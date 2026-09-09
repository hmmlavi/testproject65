"""Voice pipeline: orchestrates press-to-talk turns AND wake-word turns.

Press-to-talk flow (Phase 3):
    Talk button -> mic open (LISTENING) -> end-of-speech detected ->
    STT (faster-whisper, local) -> transcript -> SHARED brain path (the
    exact same _run_turn that text chat uses: command detection -> AI ->
    transcript) -> TTS (Fish -> local fallback) -> speakers (SPEAKING)
    -> back to idle.

Wake-word flow (Phase 4, opt-in, local-only):
    user says "Hey/Hi/Hello Gojo" while GOJO sleeps -> local detector
    (voice/wakeword.py) -> GOJO wakes -> short beep + "Gojo? Bolo." ->
    listens for the command -> SAME shared brain path -> speaks the
    reply -> sleeps again and the local listener resumes.

Rules:
- runs on worker threads; the UI polls status (no UI-thread blocking)
- every failure becomes a friendly one-line note (never a stack trace)
- the microphone is only open while GOJO is listening (press-to-talk or
  wake-listening) — and wake-listening only when the user opted in
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from ..core.state import AppState
from ..prefs import Prefs
from .playback import AudioPlayer
from .recorder import VoiceRecorder
from .stt import STTError, WhisperSTT
from .tts.base import TTSError
from .tts.normalize import normalize_for_speech
from .tts.router import TTSRouter
from .wakeword import WakeListener, WakeWordEngine, build_wakeword_engine

logger = logging.getLogger("gojo.voice.pipeline")


def _default_ack() -> None:
    """Short local beep (Windows). Never raises — the visual note alone
    is a valid acknowledgement."""
    try:
        import winsound

        winsound.Beep(880, 150)
    except Exception:  # noqa: BLE001 — non-Windows / no audio device
        pass


class VoicePipeline:
    def __init__(
        self,
        state,
        recorder: VoiceRecorder,
        stt: WhisperSTT,
        tts: TTSRouter,
        player: AudioPlayer,
        on_text,
        prefs: Prefs,
        on_note,
        audio_dir: Path,
        wake_engine: Optional[WakeWordEngine] = None,
        wake_stream_factory: Optional[Callable[[], object]] = None,
        ack_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self._state = state
        self._recorder = recorder
        self._stt = stt
        self._tts = tts
        self._player = player
        self._on_text = on_text  # GojoAPI._run_turn(text, via=...) -> dict
        self._prefs = prefs
        self._on_note = on_note  # (kind: "info"|"error", message: str)
        self._audio_dir = Path(audio_dir)
        self._lock = threading.Lock()
        self._running = False
        self._ack_fn = ack_fn or _default_ack
        self._mic_warned = False
        # Phase 4: local wake listener (off until the user opts in)
        self._listener = WakeListener(
            wake_engine or build_wakeword_engine(),
            on_wake=self._on_wake,
            stream_factory=wake_stream_factory,
        )

    @property
    def tts(self) -> TTSRouter:
        return self._tts

    @property
    def player_available(self) -> bool:
        return self._player.available

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._running

    @property
    def listener_running(self) -> bool:
        return self._listener.running

    # ------------------------------------------------------------------
    # controls (called by the UI bridge)
    # ------------------------------------------------------------------
    def start_talk(self) -> dict:
        with self._lock:
            if self._running:
                return {"ok": False, "error": "GOJO is already handling a voice turn."}
        if self._state.state is not AppState.ACTIVE:
            return {"ok": False, "error": "GOJO is not awake. Press power first."}
        if not self._player.available:
            return {
                "ok": False,
                "error": "No speakers available on this platform — voice off.",
            }
        self._state.begin_listening()
        self._recorder.start()
        with self._lock:
            self._running = True
        threading.Thread(target=self._turn, daemon=True).start()
        return {"ok": True}

    def stop_talk(self) -> dict:
        if not self._busy_check():
            return {"ok": False, "error": "Not recording."}
        self._recorder.request_stop()
        return {"ok": True}

    def _busy_check(self) -> bool:
        with self._lock:
            return self._running

    # ------------------------------------------------------------------
    # Phase 4: wake word (local, opt-in)
    # ------------------------------------------------------------------
    def set_always_listening(self, on: bool) -> None:
        """Persist the opt-in and (re)start/stop the local listener.

        Raises ValueError for bad values (the UI surfaces it).
        """
        self._prefs.set("always_listening", bool(on))
        self.sync_wake()

    def sync_wake(self) -> None:
        """Start/stop the local listener to match current settings + state.

        The listener runs ONLY when: always-listening is ON and GOJO is
        SLEEPING (waking is the whole point; an awake GOJO saves the CPU).
        """
        want = bool(self._prefs.get("always_listening")) and self._state.state is AppState.SLEEPING
        if want and not self.listener_running:
            r = self._listener.start()
            if r.get("ok"):
                self._mic_warned = False
            elif not self._mic_warned:
                self._mic_warned = True
                self._note("error", "Boss, mic nahi mil raha — wake word on nahi ho paaya.")
        elif not want and self.listener_running:
            self._listener.stop()

    def _on_wake(self, phrase: str) -> None:
        """Runs on the listener thread when a wake phrase is detected."""
        if self._state.state is not AppState.SLEEPING:
            return  # raced with a manual wake — ignore
        with self._lock:
            if self._running:
                return  # a turn is already in progress — ignore
        logger.info("wake word detected: %r", phrase)
        self._state.wake()  # SLEEPING -> ACTIVE
        self._ack()  # short local beep + visual note (never network, mic closed)
        # open the mic for the command AFTER the beep, so the beep itself
        # is never captured
        self._state.begin_listening()  # ACTIVE -> LISTENING
        self._recorder.start()
        with self._lock:
            self._running = True
        # The turn runs on its OWN thread (same as press-to-talk): the
        # listener thread below is about to exit, and re-arming the
        # listener happens after the turn — if the turn ran on the
        # listener thread, sync_wake() would see it still alive and
        # never start a fresh listener.
        threading.Thread(
            target=self._turn,
            kwargs={"after_turn": self._after_wake_turn},
            daemon=True,
        ).start()

    def _ack(self) -> None:
        self._note("info", "Gojo? Bolo.")
        try:
            self._ack_fn()
        except Exception:  # noqa: BLE001 — a failed beep must not kill the turn
            logger.exception("ack beep failed")

    def _after_wake_turn(self) -> None:
        """Wake-word contract: one turn per wake, then sleep — and the
        local listener resumes (it only re-arms if still opted in)."""
        if self._state.state is not AppState.SLEEPING:
            self._state.sleep()
        self.sync_wake()

    # ------------------------------------------------------------------
    # the turn (worker thread) — shared by press-to-talk AND wake turns
    # ------------------------------------------------------------------
    def _note(self, kind: str, message: str) -> None:
        try:
            self._on_note(kind, message)
        except Exception:  # noqa: BLE001 — a broken note hook must not kill the turn
            logger.exception("note hook failed")

    def _turn(self, after_turn: Optional[Callable[[], None]] = None) -> None:
        try:
            result = self._recorder.wait_result(timeout=180)
            if result is None or not result.ok or result.audio is None:
                self._state.stop_listening()
                if result is not None and result.error and "No microphone" in result.error:
                    self._note("error", "Boss, mic nahi mil raha. Mic check karo.")
                else:
                    self._note("error", "Recording fail ho gayi — ek baar phir try karo.")
                return

            try:
                text = self._stt.transcribe(result.audio)
            except STTError:
                logger.exception("STT failed")
                self._state.stop_listening()
                self._note("error", "I didn't catch that. Ek baar phir bolo.")
                return

            if not text:
                self._state.stop_listening()
                self._note("error", "I didn't catch that. Ek baar phir bolo.")
                return

            # Speech capture is over — leave LISTENING before the brain path
            # (so e.g. a spoken "gojo, sleep" sleeps immediately instead of
            # staying pending behind a stuck LISTENING state).
            self._state.stop_listening()

            # SHARED brain path — voice and text both end up here
            out = self._on_text(text, via="voice")
            reply = out.get("reply", "")

            if (
                self._prefs.get("tts_autoplay")
                and self._state.state is AppState.ACTIVE
                and reply
            ):
                self._speak(reply)
        except Exception:  # noqa: BLE001 — a crash here must not kill the app
            logger.exception("voice turn crashed")
            self._state.flash_error()
            self._note("error", "Kuch gadbad ho gayi, boss. Ek baar phir try karo.")
        finally:
            with self._lock:
                self._running = False
            self._state.clear_error()
            if after_turn is not None:
                try:
                    after_turn()
                except Exception:  # noqa: BLE001
                    logger.exception("after-turn hook failed")

    def _speak(self, reply: str) -> None:
        self._state.begin_speaking()
        try:
            speakable = normalize_for_speech(reply)
            try:
                result = self._tts.synthesize(speakable, self._audio_dir)
            except TTSError as exc:
                logger.error("TTS failed: %s", exc)
                self._note("error", "Aaj koi bhi voice available nahi hai — reply screen par hai.")
                return
            if result.fell_back:
                self._note(
                    "info",
                    "Fish wala voice abhi respond nahi kar raha — backup voice use kar raha hoon.",
                )
            played = self._player.play(result.path)
            if not played:
                self._note("error", "Speaker pe audio play nahi hua — reply screen par hai.")
        finally:
            self._state.end_speaking()

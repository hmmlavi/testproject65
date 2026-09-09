"""Voice pipeline (Phase 3): orchestrates the press-to-talk turn.

Flow (spec):
    Talk button -> mic open (LISTENING) -> end-of-speech detected ->
    STT (faster-whisper, local) -> transcript -> SHARED brain path (the
    exact same _run_turn that text chat uses: command detection -> AI ->
    transcript) -> TTS (Fish -> local fallback) -> speakers (SPEAKING)
    -> back to idle.

Rules:
- runs on worker threads; the UI polls status (no UI-thread blocking)
- every failure becomes a friendly one-line note (never a stack trace)
- the microphone is only ever open during an explicit press-to-talk turn
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..core.state import AppState
from ..prefs import Prefs
from .playback import AudioPlayer
from .recorder import VoiceRecorder
from .stt import STTError, WhisperSTT
from .tts.base import TTSError
from .tts.normalize import normalize_for_speech
from .tts.router import TTSRouter

logger = logging.getLogger("gojo.voice.pipeline")


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
    # the turn (worker thread)
    # ------------------------------------------------------------------
    def _note(self, kind: str, message: str) -> None:
        try:
            self._on_note(kind, message)
        except Exception:  # noqa: BLE001 — a broken note hook must not kill the turn
            logger.exception("note hook failed")

    def _turn(self) -> None:
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

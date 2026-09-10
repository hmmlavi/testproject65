"""The pywebview JS <-> Python bridge.

Every capability the UI has goes through this class: small methods, plain
JSON-serializable dict results, no window required for the logic — which
is why the whole thing is testable headless (see tests/test_api.py and
tests/test_app_wiring.py).

Rules:
- The UI never talks to the brain directly; it calls chat() / the voice
  pipeline, and renders the result.
- Destructive operations (clear_data) must be confirmed by the UI first.
- This bridge is transport-agnostic. If a later phase needs a real HTTP
  API (e.g. the Android companion), it is added on top of the same core —
  nothing here assumes an HTTP transport.
"""
from __future__ import annotations

import logging
import threading

from ..ai.base import AIProvider, ChatMessage
from ..ai.prompts import GOJO_SYSTEM_PROMPT
from ..config import Settings
from ..core.commands import detect_direct_command
from ..core.state import AppState, StateManager
from ..prefs import Prefs
from ..transcript import TranscriptStore

logger = logging.getLogger("gojo.ui.api")

# Canned acks for direct commands. These are system acknowledgements, not
# AI-generated text — that's fine and honest: a "sleep" command doesn't
# need the brain.
SLEEP_ACK = "Ja raha hoon, boss. Power dabao ya 'wake up' bolo — main wapas aa jaunga."
WAKE_ACK = "Main hoon. Kya karna hai?"
LISTEN_ON_ACK = (
    "Theek hai boss, ab main hamesha sunta rahunga. 'Hey Gojo' bolo — "
    "main turant aa jaunga. (Sab kuch local hai, koi audio kahin nahi jaata.)"
)
LISTEN_OFF_ACK = "Done. Ab main tab aata hoon jab aap power dabao ya likho."


class BrainError(Exception):
    """The brain failed mid-turn. The caller decides how to surface it."""


class GojoAPI:
    def __init__(
        self,
        settings: Settings,
        brain: AIProvider | None,
        state: StateManager,
        transcript: TranscriptStore,
        brain_error: str | None = None,
        version: str = "0.0.0",
        prefs: Prefs | None = None,
    ) -> None:
        self._settings = settings
        self._brain = brain
        self._brain_error = brain_error
        self._state = state
        self._transcript = transcript
        self._version = version
        self._history: list[ChatMessage] = []
        self._window = None  # attached by app.py once the window exists
        self._voice = None  # VoicePipeline, attached by app.py
        self._prefs = prefs or Prefs(
            settings.data_dir / "gojo_prefs.json",
            defaults={
                "voice_enabled": settings.voice_enabled,
                "tts_autoplay": settings.tts_autoplay,
                "tts_provider": settings.tts_provider,
            },
        )
        self._note = None
        self._note_lock = threading.Lock()

    # ------------------------------------------------------------------
    # window plumbing (no-ops headless so tests don't need a window)
    # ------------------------------------------------------------------
    def attach_window(self, window) -> None:
        self._window = window

    def attach_voice(self, pipeline) -> None:
        self._voice = pipeline

    def minimize(self) -> dict:
        if self._window is None:
            return {"ok": False, "error": "no window attached"}
        self._window.minimize()
        return {"ok": True}

    def close_app(self) -> dict:
        if self._window is None:
            return {"ok": False, "error": "no window attached"}
        self._window.destroy()  # pywebview 6.x: destroying the last window exits
        return {"ok": True}

    # ------------------------------------------------------------------
    # notes (one-line UI messages from the voice pipeline — read once)
    # ------------------------------------------------------------------
    def _set_note(self, kind: str, message: str) -> None:
        with self._note_lock:
            self._note = {"text": message, "error": kind == "error"}

    # ------------------------------------------------------------------
    # lifecycle (power button)
    # ------------------------------------------------------------------
    def wake(self) -> dict:
        self._state.wake()
        if self._voice is not None:
            self._voice.sync_wake()  # an awake GOJO doesn't need the wake mic
        return {"ok": True, "status": self._status()}

    def sleep(self) -> dict:
        self._state.sleep()
        if self._voice is not None:
            self._voice.sync_wake()  # resume local wake listening if opted in
        return {"ok": True, "status": self._status()}

    # ------------------------------------------------------------------
    # chat — text path. Voice arrives via the pipeline, into the SAME
    # _run_turn below (spec: one brain path, no second implementation).
    # ------------------------------------------------------------------
    def chat(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "Empty message."}
        if self._state.state is AppState.SLEEPING:
            return {"ok": False, "error": "GOJO is sleeping. Press power to wake me first."}
        try:
            out = self._run_turn(text, via="text")
        except BrainError as exc:
            return {"ok": False, "error": str(exc), "status": self._status()}
        return {"ok": True, **out, "status": self._status()}

    def _run_turn(self, text: str, via: str = "text") -> dict:
        """The shared brain path for text AND voice.

        Returns {"kind": "reply"|"command", "reply": str}.
        Raises BrainError if the brain fails (caller surfaces it).
        """
        # Direct commands short-circuit the brain (cheap, instant, real) —
        # including when spoken: "gojo, sleep" works by voice too.
        command = detect_direct_command(text)
        if command and self._state.state is AppState.LISTENING:
            self._state.stop_listening()  # safety net for command path
        if command in ("listening_on", "listening_off"):
            # Phase 4: toggle the LOCAL wake-word listener. Persists in
            # prefs, applies live; no brain, no network.
            on = command == "listening_on"
            try:
                self._prefs.set("always_listening", on)
            except ValueError:
                return {"kind": "command", "reply": "Boss, yeh setting change nahi ho paayi."}
            if self._voice is not None:
                self._voice.sync_wake()
            ack = LISTEN_ON_ACK if on else LISTEN_OFF_ACK
            self._transcript.append("user", text, via=via)
            self._transcript.append("model", ack, via="system")
            return {"kind": "command", "reply": ack}
        if command == "sleep":
            self._transcript.append("user", text, via=via)
            self._transcript.append("model", SLEEP_ACK, via="system")
            self._state.sleep()
            if self._voice is not None:
                self._voice.sync_wake()  # resume local wake listening if opted in
            return {"kind": "command", "reply": SLEEP_ACK}
        if command == "wake":
            # We're already awake here (sleeping was blocked by the caller).
            return {"kind": "command", "reply": WAKE_ACK}

        if self._brain is None:
            raise BrainError(self._brain_error or "No AI brain configured.")

        # Diagnostic line: facts only — never the conversation content.
        logger.info("brain: via=%s input=%d chars (model=%s)",
                    via, len(text), self._settings.gemini_model)
        self._state.begin_thinking()
        self._history.append(ChatMessage(role="user", text=text))
        try:
            reply = self._brain.chat(self._history, GOJO_SYSTEM_PROMPT)
        except Exception as exc:  # surface to the caller; never crash the app
            self._history.pop()  # a failed turn must not poison the history
            self._state.end_thinking()
            raise BrainError(f"Brain error: {exc}") from exc

        logger.info("brain: reply=%d chars", len(reply))
        self._history.append(ChatMessage(role="model", text=reply))
        self._transcript.append("user", text, via=via)
        self._transcript.append("model", reply, via="ai")
        self._state.end_thinking()  # applies any pending sleep here
        return {"kind": "reply", "reply": reply}

    # ------------------------------------------------------------------
    # voice (Phase 3) — thin pass-through to the pipeline
    # ------------------------------------------------------------------
    def start_talk(self) -> dict:
        if self._voice is None:
            return {"ok": False, "error": "Voice system not initialized."}
        if not (self._prefs.get("voice_enabled") and self._settings.voice_enabled):
            return {"ok": False, "error": "Voice is turned off in Settings."}
        return self._voice.start_talk()

    def stop_talk(self) -> dict:
        if self._voice is None:
            return {"ok": False, "error": "Voice system not initialized."}
        return self._voice.stop_talk()

    def test_mic(self) -> dict:
        """Debug: which device does GOJO use, and are samples arriving?
        Reads ~1.2 s of level-only stats; nothing is recorded or stored."""
        if self._voice is None:
            return {"ok": False, "error": "Voice system not initialized."}
        return self._voice.test_mic()

    # ------------------------------------------------------------------
    # transcript
    # ------------------------------------------------------------------
    def get_transcript(self) -> dict:
        return {"ok": True, "items": self._transcript.load()}

    def new_chat(self) -> dict:
        self._transcript.new_session()
        self._history = []
        return {"ok": True, "items": []}

    def note_history(self, role: str, text: str) -> None:
        """Seed the AI history from a loaded transcript (app startup)."""
        self._history.append(ChatMessage(role=role, text=text))

    # ------------------------------------------------------------------
    # status / settings
    # ------------------------------------------------------------------
    def _status(self) -> dict:
        with self._note_lock:
            note = self._note
            self._note = None  # read-once
        return {
            "state": self._state.state.value,
            "provider": self._settings.provider,
            "model": self._settings.gemini_model,
            "busy": self._voice.busy if self._voice is not None else False,
            "wake_listening": self._voice.listener_running if self._voice is not None else False,
            "note": note,
        }

    def get_status(self) -> dict:
        return {"ok": True, "status": self._status()}

    def get_settings(self) -> dict:
        return {
            "ok": True,
            "provider": self._settings.provider,
            "model": self._settings.gemini_model,
            "data_dir": str(self._settings.data_dir),
            "version": self._version,
        }

    def get_voice_settings(self) -> dict:
        s = self._settings
        fish_key = s.fish_api_key or ""
        return {
            "ok": True,
            "voice_enabled": bool(self._prefs.get("voice_enabled") and s.voice_enabled),
            "tts_autoplay": bool(self._prefs.get("tts_autoplay")),
            "tts_provider": self._prefs.get("tts_provider"),
            "always_listening": bool(self._prefs.get("always_listening")),
            "wake_listening": self._voice.listener_running if self._voice else False,
            "mic": self._voice.mic_info() if self._voice else {
                "configured": s.mic_device or "system default",
                "in_use": "not opened yet",
                "listener_frames": 0,
            },
            "fish_configured": bool(
                fish_key and not fish_key.strip().lower().startswith("paste-")
            ),
            "fish_voice": s.fish_model_id,
            "stt_model": s.stt_model_size,
            "player_available": self._voice.player_available if self._voice else False,
        }

    def set_voice_pref(self, key: str, value) -> dict:
        try:
            prefs = self._prefs.set(key, value)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        # apply live where it matters
        if key == "tts_provider" and self._voice is not None:
            try:
                self._voice.tts.set_order(value)
            except ValueError:
                return {"ok": False, "error": f"Unknown TTS provider: {value!r}"}
        if key == "always_listening" and self._voice is not None:
            self._voice.sync_wake()
        return {"ok": True, "prefs": prefs}

    # ------------------------------------------------------------------
    # destructive — the UI must confirm before calling
    # ------------------------------------------------------------------
    def clear_data(self) -> dict:
        self._transcript.clear_all()
        self._history = []
        return {"ok": True}

"""The pywebview JS <-> Python bridge.

Every capability the UI has goes through this class: small methods, plain
JSON-serializable dict results, no window required for the logic — which
is why the whole thing is testable headless (see tests/test_api.py).

Rules:
- The UI never talks to the brain directly; it calls chat() and renders.
- Destructive operations (clear_data) must be confirmed by the UI first.
- This bridge is transport-agnostic. If a later phase needs a real HTTP
  API (e.g. the Android companion), it is added on top of the same core —
  nothing here assumes an HTTP transport.
"""
from __future__ import annotations

from ..ai.base import AIProvider, ChatMessage
from ..ai.prompts import GOJO_SYSTEM_PROMPT
from ..config import Settings
from ..core.commands import detect_direct_command
from ..core.state import AppState, StateManager
from ..transcript import TranscriptStore

# Canned acks for direct commands. These are system acknowledgements, not
# AI-generated text — that's fine and honest: a "sleep" command doesn't
# need the brain.
SLEEP_ACK = "Ja raha hoon, boss. Power dabao ya 'wake up' bolo — main wapas aa jaunga."
WAKE_ACK = "Main hoon. Kya karna hai?"


class GojoAPI:
    def __init__(
        self,
        settings: Settings,
        brain: AIProvider | None,
        state: StateManager,
        transcript: TranscriptStore,
        brain_error: str | None = None,
        version: str = "0.0.0",
    ) -> None:
        self._settings = settings
        self._brain = brain
        self._brain_error = brain_error
        self._state = state
        self._transcript = transcript
        self._version = version
        self._history: list[ChatMessage] = []
        self._window = None  # attached by app.py once the window exists

    # ------------------------------------------------------------------
    # window plumbing (no-ops headless so tests don't need a window)
    # ------------------------------------------------------------------
    def attach_window(self, window) -> None:
        self._window = window

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
    # lifecycle (power button)
    # ------------------------------------------------------------------
    def wake(self) -> dict:
        self._state.wake()
        return {"ok": True, "status": self._status()}

    def sleep(self) -> dict:
        self._state.sleep()
        return {"ok": True, "status": self._status()}

    # ------------------------------------------------------------------
    # chat — the main path. UI sends text, gets a renderable result.
    # ------------------------------------------------------------------
    def chat(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "Empty message."}

        if self._state.state is AppState.SLEEPING:
            return {"ok": False, "error": "GOJO is sleeping. Press power to wake me first."}

        # Direct commands short-circuit the brain (cheap, instant, real).
        command = detect_direct_command(text)
        if command == "sleep":
            self._transcript.append("user", text)
            self._transcript.append("model", SLEEP_ACK)
            self._state.sleep()
            return {"ok": True, "kind": "command", "reply": SLEEP_ACK, "status": self._status()}
        if command == "wake":
            # We're already awake here (sleeping was blocked above).
            return {"ok": True, "kind": "command", "reply": WAKE_ACK, "status": self._status()}

        if self._brain is None:
            return {"ok": False, "error": self._brain_error or "No AI brain configured."}

        self._state.begin_thinking()
        self._history.append(ChatMessage(role="user", text=text))
        try:
            reply = self._brain.chat(self._history, GOJO_SYSTEM_PROMPT)
        except Exception as exc:  # surface to the UI; never crash the app
            self._history.pop()  # a failed turn must not poison the history
            self._state.end_thinking()
            return {"ok": False, "error": f"Brain error: {exc}", "status": self._status()}

        self._history.append(ChatMessage(role="model", text=reply))
        self._transcript.append("user", text)
        self._transcript.append("model", reply)
        self._state.end_thinking()  # applies any pending sleep here
        return {"ok": True, "kind": "reply", "reply": reply, "status": self._status()}

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
        return {
            "state": self._state.state.value,
            "provider": self._settings.provider,
            "model": self._settings.gemini_model,
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

    # ------------------------------------------------------------------
    # destructive — the UI must confirm before calling
    # ------------------------------------------------------------------
    def clear_data(self) -> dict:
        self._transcript.clear_all()
        self._history = []
        return {"ok": True}

"""GOJO state machine — where GOJO is in the world right now.

States:
    SLEEPING   — GOJO is down. Input disabled, avatar dimmed.
                 App boots into SLEEPING (spec: starts with PC in a sleep
                 state and only activates on request).
    ACTIVE     — awake, ready to respond.
    LISTENING  — microphone open, capturing speech (Phase 3, press-to-talk).
    THINKING   — processing a request (transient).
    SPEAKING   — TTS audio playing (Phase 3).
    ERROR      — transient failure flag; always auto-cleared to ACTIVE.

Phase 4 wake-word listening does NOT add a state: "wake listening" is
derived in the UI as (state == SLEEPING AND local listener running) —
continuous listening stays off unless the user explicitly enables it.

Thread-safe: the UI bridge and the voice pipeline call this from other
threads while chat processing runs.
"""
from __future__ import annotations

import threading
from enum import Enum
from typing import Callable


class AppState(str, Enum):
    SLEEPING = "sleeping"
    ACTIVE = "active"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class StateManager:
    """Small, explicit state machine. No hidden transitions."""

    def __init__(self, initial: AppState = AppState.SLEEPING) -> None:
        self._state = initial
        self._lock = threading.Lock()
        self._listeners: list[Callable[[AppState], None]] = []
        self._pending_sleep = False

    # -- introspection -------------------------------------------------
    @property
    def state(self) -> AppState:
        with self._lock:
            return self._state

    @property
    def pending_sleep(self) -> bool:
        """True when sleep was requested during a busy state and will apply
        as soon as the current work finishes."""
        with self._lock:
            return self._pending_sleep

    # -- listeners (UI uses these later; polling also works today) ------
    def add_listener(self, fn: Callable[[AppState], None]) -> None:
        self._listeners.append(fn)

    def _set(self, new: AppState) -> bool:
        """Set state, notify listeners. Returns True if it changed."""
        with self._lock:
            if self._state == new:
                return False
            self._state = new
        for fn in list(self._listeners):
            try:
                fn(new)
            except Exception:  # a broken listener must not break GOJO
                pass
        return True

    # -- lifecycle ------------------------------------------------------
    def wake(self) -> AppState:
        """Anything -> ACTIVE. Clears pending sleep."""
        with self._lock:
            self._pending_sleep = False
        self._set(AppState.ACTIVE)
        return self.state

    def sleep(self) -> AppState:
        """Go to SLEEPING.

        If busy (LISTENING / THINKING / SPEAKING), the sleep is *pending*:
        it applies automatically when the current work finishes (GOJO never
        cuts itself off mid-thought or mid-sentence).
        """
        if self.state in (AppState.LISTENING, AppState.THINKING, AppState.SPEAKING):
            with self._lock:
                self._pending_sleep = True
            return self.state
        with self._lock:
            self._pending_sleep = False
        self._set(AppState.SLEEPING)
        return self.state

    # -- voice states (Phase 3) ------------------------------------------
    def begin_listening(self) -> AppState:
        """ACTIVE -> LISTENING. Only from ACTIVE (sleeping must wake first)."""
        if self.state is AppState.ACTIVE:
            self._set(AppState.LISTENING)
        return self.state

    def stop_listening(self) -> AppState:
        """LISTENING -> ACTIVE (recording cancelled / nothing captured)."""
        if self.state is AppState.LISTENING:
            self._set(AppState.ACTIVE)
        return self.state

    def begin_speaking(self) -> AppState:
        """ACTIVE -> SPEAKING (TTS audio is playing)."""
        if self.state is AppState.ACTIVE:
            self._set(AppState.SPEAKING)
        return self.state

    def end_speaking(self) -> AppState:
        """SPEAKING -> SLEEPING (if sleep was pending) else ACTIVE."""
        if self.state is not AppState.SPEAKING:
            return self.state
        with self._lock:
            pending = self._pending_sleep
            self._pending_sleep = False
        self._set(AppState.SLEEPING if pending else AppState.ACTIVE)
        return self.state

    # -- thinking ---------------------------------------------------------
    def begin_thinking(self) -> AppState:
        """ACTIVE or LISTENING -> THINKING. Sleeping must wake first."""
        if self.state in (AppState.ACTIVE, AppState.LISTENING):
            self._set(AppState.THINKING)
        return self.state

    def end_thinking(self) -> AppState:
        """THINKING -> SLEEPING (if sleep was pending) else ACTIVE."""
        if self.state is not AppState.THINKING:
            return self.state
        with self._lock:
            pending = self._pending_sleep
            self._pending_sleep = False
        self._set(AppState.SLEEPING if pending else AppState.ACTIVE)
        return self.state

    # -- error (transient) ------------------------------------------------
    def flash_error(self) -> AppState:
        """Mark a visible error. The pipeline always clears it afterwards."""
        if self.state not in (AppState.SLEEPING, AppState.ERROR):
            self._set(AppState.ERROR)
        return self.state

    def clear_error(self) -> AppState:
        if self.state is AppState.ERROR:
            self._set(AppState.ACTIVE)
        return self.state

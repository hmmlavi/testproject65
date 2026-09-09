"""GOJO state machine — where GOJO is in the world right now.

States (Phase 2):
    SLEEPING  — GOJO is down. Input is disabled, avatar is dimmed.
                App boots into SLEEPING (spec: starts with PC in a sleep
                state and only activates on request).
    ACTIVE    — awake, ready to respond.
    THINKING  — processing a request (transient; always returns to
                ACTIVE or SLEEPING).

Phase 3 will add LISTENING (mic open) and SPEAKING (voice out) as new
members of this same enum — no other code changes required for that.

Thread-safe: the UI bridge calls this from another thread while chat
processing runs.
"""
from __future__ import annotations

import threading
from enum import Enum
from typing import Callable


class AppState(str, Enum):
    SLEEPING = "sleeping"
    ACTIVE = "active"
    THINKING = "thinking"


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
        """True when sleep was requested during THINKING and will apply
        as soon as thinking ends."""
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

    # -- transitions ----------------------------------------------------
    def wake(self) -> AppState:
        """SLEEPING -> ACTIVE (no-op if already awake). Clears pending sleep."""
        with self._lock:
            self._pending_sleep = False
        self._set(AppState.ACTIVE)
        return self.state

    def sleep(self) -> AppState:
        """ACTIVE -> SLEEPING.

        If THINKING, the sleep is *pending*: it applies automatically when
        the current response finishes (GOJO finishes its sentence, then
        goes down — never cut off mid-thought).
        """
        if self.state is AppState.THINKING:
            with self._lock:
                self._pending_sleep = True
            return self.state
        with self._lock:
            self._pending_sleep = False
        self._set(AppState.SLEEPING)
        return self.state

    def begin_thinking(self) -> AppState:
        """ACTIVE -> THINKING. Only from ACTIVE (sleeping must wake first)."""
        if self.state is AppState.ACTIVE:
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

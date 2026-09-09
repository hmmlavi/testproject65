"""Direct-command detection — tiny, honest, tested.

Recognized from short imperative messages (typed OR transcribed speech):
    "sleep"                -> GOJO goes to sleep
    "wake"                 -> GOJO wakes up
    "always listening on"  -> local wake-word detection ON  (Phase 4)
    "always listening off" -> local wake-word detection OFF (Phase 4)
                             ("wake word on/off" is an alias)

This is deliberately a pattern matcher, not an NLU: small surface area,
zero fake intelligence.
"""
from __future__ import annotations

import re

# Keep only short imperatives so "sleep" inside a long paragraph is ignored.
_MAX_COMMAND_LEN = 40

_SLEEP_PATTERNS = (
    r"\bgo to sleep\b",
    r"\bsleep\b",
    r"\bso ja\b",
    r"\bso jao\b",
    r"\bso jaana\b",
    r"\bsona\b",
)

_WAKE_PATTERNS = (
    r"\bwake ?up\b",
    r"\bawake\b",
    r"\bjaggo\b",
    r"\bjaago\b",
    r"\butho\b",
)

# Phase 4: checked FIRST — "wake word on" contains "wake" and must not be
# parsed as the plain wake command.
_LISTENING_PATTERNS = (
    r"\balways listening\b",
    r"\bwake ?word\b",
)


def detect_direct_command(text: str) -> str | None:
    """Return "sleep" | "wake" | "listening_on" | "listening_off" | None
    for short imperative messages."""
    t = (text or "").lower().strip()
    if not t or len(t) > _MAX_COMMAND_LEN:
        return None
    # strip punctuation so "gojo, sleep." matches "\bsleep\b"
    t = re.sub(r"[^\w\s]", " ", t)

    for pattern in _LISTENING_PATTERNS:
        if re.search(pattern, t):
            has_on = bool(re.search(r"\bon\b", t))
            has_off = bool(re.search(r"\boff\b", t))
            if has_on and not has_off:
                return "listening_on"
            if has_off and not has_on:
                return "listening_off"
            return None  # ambiguous ("...on off") — don't guess
    for pattern in _SLEEP_PATTERNS:
        if re.search(pattern, t):
            return "sleep"
    for pattern in _WAKE_PATTERNS:
        if re.search(pattern, t):
            return "wake"
    return None

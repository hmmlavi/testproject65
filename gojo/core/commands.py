"""Direct-command detection — tiny, honest, tested.

Phase 2 recognizes only lifecycle commands from typed text:
    "sleep"  -> GOJO goes to sleep
    "wake"   -> GOJO wakes up

This is deliberately a pattern matcher, not an NLU: small surface area,
zero fake intelligence. It is the seed of the command layer the voice
phase will reuse — the same function, voice transcript in, command out.
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


def detect_direct_command(text: str) -> str | None:
    """Return "sleep" | "wake" | None for short imperative messages."""
    t = (text or "").lower().strip()
    if not t or len(t) > _MAX_COMMAND_LEN:
        return None
    # strip punctuation so "gojo, sleep." matches "\bsleep\b"
    t = re.sub(r"[^\w\s]", " ", t)

    for pattern in _SLEEP_PATTERNS:
        if re.search(pattern, t):
            return "sleep"
    for pattern in _WAKE_PATTERNS:
        if re.search(pattern, t):
            return "wake"
    return None

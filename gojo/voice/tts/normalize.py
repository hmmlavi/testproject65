"""Speakable-text layer for TTS (Phase 3).

Runs ONLY on the audio path. The visible transcript and the AI history keep
the AI's actual response, untouched (per spec).

Rules:
- Preserve technical terms exactly: VS Code, Python, GitHub, Gemini, API,
  Android, SaBuddy, ... (never blindly "Hindi-ize" them)
- Expand speech-hostile symbols: C++ -> "C plus plus", ₹ -> "rupees"
- Strip markdown / code formatting, drop emoji, collapse whitespace
- Optional override table for words the voice mispronounces (extend as
  needed — data, not code changes)
"""
from __future__ import annotations

import re
import unicodedata

# word (case-insensitive, word-bounded) -> spoken form.
# Empty by default: only add entries when a specific mispronunciation is
# observed. This is the "configurable pronunciation-normalization layer".
DEFAULT_OVERRIDES: dict[str, str] = {}

_MD_CHARS = re.compile(r"[*_#`>|~]+")
# NOTE: no trailing \b after "+" (a \b after a non-word char never matches)
_CPLUSPLUS = re.compile(r"\bC\+\+(?!\+)", re.IGNORECASE)
_RUPEE = re.compile(r"₹|\bRs\.?|\bINR\b", re.IGNORECASE)
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200d]"
)
_WS = re.compile(r"\s+")


def normalize_for_speech(text: str, overrides: dict[str, str] | None = None) -> str:
    """AI response text -> text that reads well out loud."""
    t = unicodedata.normalize("NFKC", text or "")
    t = _MD_CHARS.sub(" ", t)
    t = _CPLUSPLUS.sub("C plus plus", t)
    t = _RUPEE.sub("rupees ", t)
    t = _EMOJI.sub(" ", t)
    for old, new in (overrides or DEFAULT_OVERRIDES).items():
        t = re.sub(rf"\b{re.escape(old)}\b", new, t, flags=re.IGNORECASE)
    t = _WS.sub(" ", t).strip()
    return t

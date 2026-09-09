"""TTS provider interface.

Every speech engine GOJO can use implements TTSProvider. This is what makes
the voice swappable: the rest of GOJO only talks to the TTSRouter, never to
Fish Audio or Windows SAPI directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class TTSError(Exception):
    """A TTS provider failed. `retryable` hints that a fallback may succeed."""

    def __init__(self, message: str, provider: str | None = None, retryable: bool = True) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class TTSProvider:
    """Interface that all TTS providers implement."""

    name: str = "base"

    @property
    def available(self) -> bool:
        """False when this provider cannot work here (missing key, missing
        platform/dependency). The router skips unavailable providers."""
        return True

    def synthesize(self, text: str, out_dir: Path) -> Path:
        """Render `text` to an audio file inside `out_dir`.

        The provider chooses its own filename + extension (.mp3 / .wav).
        Returns the written path. Raises TTSError on any failure — never
        leaves a half-written file presented as success.
        """
        raise NotImplementedError


@dataclass
class TTSResult:
    path: Path  # the audio file actually written
    provider: str  # which provider produced it
    fell_back: bool  # True when a fallback provider was used

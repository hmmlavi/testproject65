"""TTS router — provider chain with automatic fallback.

Order comes from the user's preference (tts_provider):
    "auto"    -> [fish, windows]   (fish primary, local fallback)
    "fish"    -> [fish, windows]   (fish first; fallback still protects us)
    "windows" -> [windows]         (fully local)

Per spec: if Fish Audio has no key, is offline, rate-limits, or returns a
bad payload, GOJO does NOT crash — it falls back to local TTS (and tells the
user, exactly once per failure burst).
"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import TTSError, TTSProvider, TTSResult

logger = logging.getLogger("gojo.voice.tts")


class TTSRouter:
    def __init__(self, providers: list[TTSProvider], order: str = "auto") -> None:
        self._by_name = {p.name: p for p in providers}
        if not self._by_name:
            raise ValueError("TTSRouter needs at least one provider")
        self._order = "auto"
        self.set_order(order)
        self.last_fallback_used = False  # UI reads this to show the one-time note

    def set_order(self, order: str) -> None:
        """Change the chain live (settings panel). 'auto'/'fish' keep the
        local fallback in the chain; 'windows' is fully local."""
        if order not in ("auto", "fish", "windows"):
            raise ValueError(f"Unknown TTS provider order: {order!r}")
        self._order = order
        names = ["windows"] if order == "windows" else ["fish", "windows"]
        self._chain: list[TTSProvider] = [self._by_name[n] for n in names if n in self._by_name]
        if not self._chain:
            raise ValueError("TTSRouter has no providers left for this order")

    @property
    def chain(self) -> list[str]:
        return [p.name for p in self._chain]

    def synthesize(self, text: str, out_dir: Path) -> TTSResult:
        """Try each provider in the chain; return the first success.

        Raises TTSError (with all attempts listed) only when every provider
        failed — the pipeline turns that into a friendly spoken/text message.
        """
        attempts: list[str] = []
        for index, provider in enumerate(self._chain):
            if not provider.available:
                attempts.append(f"{provider.name}: unavailable")
                continue
            try:
                path = provider.synthesize(text, out_dir)
                self.last_fallback_used = index > 0
                if index > 0:
                    logger.warning("TTS fell back to %s", provider.name)
                return TTSResult(path=path, provider=provider.name, fell_back=index > 0)
            except TTSError as exc:
                attempts.append(f"{provider.name}: {exc}")
                continue
            except Exception as exc:  # a provider bug must not kill the voice path
                attempts.append(f"{provider.name}: unexpected error: {exc}")
                logger.exception("TTS provider %s crashed", provider.name)
        raise TTSError("All TTS providers failed — " + " | ".join(attempts))

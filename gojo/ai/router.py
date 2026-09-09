"""AI router — decides which brain handles the request.

Phase 1: a single provider chosen from config (GOJO_PROVIDER in .env).
The structure is already in place so later phases can add real routing rules
(e.g. coding tasks -> coding model, private/offline tasks -> local Ollama
model, quick chat -> fast model) without changing anything outside this file.
"""
from __future__ import annotations

from ..config import Settings
from .base import AIProvider


def build_router(settings: Settings) -> AIProvider:
    """Build the AI provider selected in config.

    Raises:
        RuntimeError: if the selected provider is misconfigured (e.g. missing key).
        ValueError: if the provider name is unknown.
    """
    provider = (settings.provider or "gemini").strip().lower()

    if provider == "gemini":
        from .gemini import GeminiProvider  # lazy import: mock mode needs no SDK

        return GeminiProvider(
            api_key=settings.gemini_api_key, model=settings.gemini_model
        )

    if provider == "mock":
        from .mock import MockProvider

        return MockProvider()

    raise ValueError(
        f"Unknown GOJO_PROVIDER: {settings.provider!r}. "
        "Use 'gemini' (real brain) or 'mock' (offline wiring test)."
    )

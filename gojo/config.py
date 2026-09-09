"""GOJO configuration.

All settings come from environment variables, normally loaded from the local
`.env` file next to this project. Secrets (API keys) live ONLY in `.env`,
which is git-ignored and never committed.

Runtime-flippable preferences (voice on/off, autoplay, TTS provider) are
seeded from here and then stored in data/gojo_prefs.json (see gojo/prefs.py)
— that's what the settings panel writes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# The GOJO folder (this file lives in <GOJO>/gojo/config.py)
ROOT_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "true" if default else "false").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # --- core (Phase 1) ---
    gemini_api_key: str | None
    gemini_model: str
    provider: str  # "gemini" (real brain) | "mock" (offline wiring test)
    data_dir: Path  # local data (memory DB, caches) — used from Phase 6
    log_level: str
    # --- voice (Phase 3) — defaults; UI overrides live in Prefs ---
    voice_enabled: bool = True
    tts_autoplay: bool = True
    tts_provider: str = "auto"  # "auto" | "fish" | "windows"
    fish_api_key: str | None = None
    fish_model_id: str = "0a7bf4b832b2459eb012547b4e3643d3"
    fish_model: str = "s2.1-pro-free"  # goes in the "model" HTTP header
    stt_model_size: str = "small"  # whisper: tiny | base | small (CPU-friendly)


def load_settings() -> Settings:
    """Load settings from environment / .env. Safe to call multiple times."""
    load_dotenv(ROOT_DIR / ".env")
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        provider=os.getenv("GOJO_PROVIDER", "gemini"),
        data_dir=ROOT_DIR / "data",
        log_level=os.getenv("GOJO_LOG_LEVEL", "INFO"),
        voice_enabled=_env_bool("GOJO_VOICE_ENABLED", True),
        tts_autoplay=_env_bool("GOJO_TTS_AUTOPLAY", True),
        tts_provider=os.getenv("GOJO_TTS_PROVIDER", "auto").strip().lower(),
        fish_api_key=os.getenv("FISH_AUDIO_API_KEY"),
        fish_model_id=os.getenv("FISH_AUDIO_MODEL_ID", "0a7bf4b832b2459eb012547b4e3643d3").strip(),
        fish_model=os.getenv("FISH_AUDIO_MODEL", "s2.1-pro-free").strip(),
        stt_model_size=os.getenv("GOJO_STT_MODEL", "small").strip().lower(),
    )

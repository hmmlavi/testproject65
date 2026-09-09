"""GOJO configuration.

All settings come from environment variables, normally loaded from the local
`.env` file next to this project. Secrets (API keys) live ONLY in `.env`,
which is git-ignored and never committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# The GOJO folder (this file lives in <GOJO>/gojo/config.py)
ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    provider: str  # "gemini" (real brain) | "mock" (offline wiring test)
    data_dir: Path  # local data (memory DB, caches) — used from Phase 6
    log_level: str


def load_settings() -> Settings:
    """Load settings from environment / .env. Safe to call multiple times."""
    load_dotenv(ROOT_DIR / ".env")
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        provider=os.getenv("GOJO_PROVIDER", "gemini"),
        data_dir=ROOT_DIR / "data",
        log_level=os.getenv("GOJO_LOG_LEVEL", "INFO"),
    )

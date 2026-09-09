"""Tiny user-preference store (data/gojo_prefs.json).

For the few settings the user can flip at runtime from the settings panel:
    voice_enabled   (bool)
    tts_autoplay    (bool)
    tts_provider    ("auto" | "fish" | "windows")

Defaults are seeded from .env (Settings); the JSON file only stores
overrides. Secrets and .env-only values (API keys, model IDs) NEVER live
here, and this file is inside data/ (git-ignored).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

VALID_PROVIDERS = ("auto", "fish", "windows")

# key: (type, default)
_PREF_SCHEMA: dict[str, tuple[type, Any]] = {
    "voice_enabled": (bool, True),
    "tts_autoplay": (bool, True),
    "tts_provider": (str, "auto"),
}


class Prefs:
    def __init__(self, path: Path | str, defaults: dict[str, Any] | None = None) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        for key, (_type, default) in _PREF_SCHEMA.items():
            self._data[key] = (defaults or {}).get(key, default)
        self._load_file()

    def _load_file(self) -> None:
        if not self._path.exists():
            return
        try:
            stored = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return  # corrupt prefs must not break GOJO — fall back to defaults
        if not isinstance(stored, dict):
            return
        for key, (ptype, _default) in _PREF_SCHEMA.items():
            value = stored.get(key)
            if self._valid(key, value):
                self._data[key] = value

    @staticmethod
    def _valid(key: str, value: Any) -> bool:
        ptype, _default = _PREF_SCHEMA[key]
        if ptype is bool:
            return isinstance(value, bool)
        if key == "tts_provider":
            return isinstance(value, str) and value in VALID_PROVIDERS
        return isinstance(value, str)

    def get(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def set(self, key: str, value: Any) -> dict[str, Any]:
        """Validate + persist a pref. Returns the new full prefs dict.

        Raises ValueError for unknown keys or invalid values — the UI
        surfaces that as a friendly error instead of corrupting state.
        """
        if key not in _PREF_SCHEMA:
            raise ValueError(f"Unknown preference: {key!r}")
        if not self._valid(key, value):
            raise ValueError(f"Invalid value for {key!r}: {value!r}")
        with self._lock:
            self._data[key] = value
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return dict(self._data)

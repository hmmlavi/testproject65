"""Checks for the runtime preference store (data/gojo_prefs.json).

Run:  python -m tests.test_prefs
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from gojo.prefs import Prefs


def _prefs(tmp: Path, defaults: dict | None = None) -> Prefs:
    return Prefs(tmp / "gojo_prefs.json", defaults=defaults)


def test_defaults_from_settings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _prefs(Path(tmp), defaults={"voice_enabled": False, "tts_provider": "fish"})
        assert p.get("voice_enabled") is False
        assert p.get("tts_provider") == "fish"
        assert p.get("tts_autoplay") is True  # schema default
    print("PASS test_defaults_from_settings")


def test_set_validates_and_persists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gojo_prefs.json"
        p = _prefs(Path(tmp))
        p.set("tts_provider", "windows")
        p.set("tts_autoplay", False)
        # persisted on disk
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["tts_provider"] == "windows"
        assert stored["tts_autoplay"] is False
        # a fresh Prefs instance reads the overrides back
        p2 = _prefs(Path(tmp))
        assert p2.get("tts_provider") == "windows"
        assert p2.get("tts_autoplay") is False
    print("PASS test_set_validates_and_persists")


def test_invalid_values_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = _prefs(Path(tmp))
        for bad in ("chromium", "", 42, ["auto"]):
            try:
                p.set("tts_provider", bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid tts_provider accepted: {bad!r}")
        assert p.get("tts_provider") == "auto"  # unchanged
        try:
            p.set("not_a_pref", "x")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown pref accepted")
        try:
            p.set("voice_enabled", "yes")  # bool required, not str
        except ValueError:
            pass
        else:
            raise AssertionError("non-bool accepted for voice_enabled")
    print("PASS test_invalid_values_rejected")


def test_corrupt_file_falls_back_to_defaults() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gojo_prefs.json"
        path.write_text("{not json!!", encoding="utf-8")
        p = _prefs(Path(tmp), defaults={"tts_provider": "fish"})
        assert p.get("tts_provider") == "fish"  # defaults survive corruption
        p.set("tts_provider", "windows")  # and can be rewritten cleanly
        assert json.loads(path.read_text(encoding="utf-8"))["tts_provider"] == "windows"
    print("PASS test_corrupt_file_falls_back_to_defaults")


def main() -> int:
    tests = [
        test_defaults_from_settings,
        test_set_validates_and_persists,
        test_invalid_values_rejected,
        test_corrupt_file_falls_back_to_defaults,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

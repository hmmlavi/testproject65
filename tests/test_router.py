"""Zero-dependency checks for the GOJO AI layer.

Run from the GOJO folder:
    python -m tests.test_router

No API key needed — these test the wiring (provider selection, config,
mock round-trip, error handling), not the live AI.
"""
from __future__ import annotations

from pathlib import Path

from gojo.ai.base import ChatMessage
from gojo.ai.prompts import GOJO_SYSTEM_PROMPT
from gojo.ai.router import build_router
from gojo.config import Settings


def _settings(provider: str, key: str | None = None) -> Settings:
    return Settings(
        gemini_api_key=key,
        gemini_model="test-model",
        provider=provider,
        data_dir=Path("data"),
        log_level="INFO",
    )


def test_mock_roundtrip() -> None:
    """Mock provider must echo the user text and clearly label itself."""
    brain = build_router(_settings("mock"))
    out = brain.chat([ChatMessage(role="user", text="hello gojo")], GOJO_SYSTEM_PROMPT)
    assert "hello gojo" in out, f"mock did not echo input: {out!r}"
    assert "MOCK" in out, "mock must clearly label itself as not real AI"
    print("PASS test_mock_roundtrip")


def test_unknown_provider_raises() -> None:
    try:
        build_router(_settings("definitely-not-a-provider"))
    except ValueError as exc:
        assert "Unknown GOJO_PROVIDER" in str(exc)
        print("PASS test_unknown_provider_raises")
        return
    raise AssertionError("unknown provider did not raise ValueError")


def test_gemini_without_key_raises_friendly_error() -> None:
    """Missing key must fail with a clear, actionable message — not a traceback."""
    try:
        build_router(_settings("gemini", key=None))
    except RuntimeError as exc:
        msg = str(exc)
        assert "GEMINI_API_KEY" in msg, f"error should mention the key: {msg!r}"
        assert "aistudio.google.com" in msg, f"error should say where to get it: {msg!r}"
        print("PASS test_gemini_without_key_raises_friendly_error")
        return
    raise AssertionError("missing key did not raise RuntimeError")


def test_personality_prompt_is_not_corporate() -> None:
    """The personality prompt must actually ban the robotic phrases."""
    for banned in ("BANNED", "How may I assist you today", "Certainly!"):
        assert banned in GOJO_SYSTEM_PROMPT, f"prompt missing guard for: {banned}"
    print("PASS test_personality_prompt_is_not_corporate")


def main() -> int:
    tests = [
        test_mock_roundtrip,
        test_unknown_provider_raises,
        test_gemini_without_key_raises_friendly_error,
        test_personality_prompt_is_not_corporate,
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

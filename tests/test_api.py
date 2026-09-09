"""Headless checks for the UI bridge (GojoAPI) — no window, no network.

Uses the clearly-labeled MockProvider as the brain, so the full chat flow
(input -> command check -> brain -> transcript -> state) is exercised
end to end without Gemini.

Run:  python -m tests.test_api
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from gojo.ai.mock import MockProvider
from gojo.config import Settings
from gojo.core.state import AppState, StateManager
from gojo.transcript import TranscriptStore
from gojo.ui.api import SLEEP_ACK, WAKE_ACK, GojoAPI


def _settings(tmp: Path) -> Settings:
    return Settings(
        gemini_api_key=None,
        gemini_model="mock-model",
        provider="mock",
        data_dir=tmp / "data",
        log_level="INFO",
    )


def _api(tmp: Path, awake: bool = True) -> GojoAPI:
    state = StateManager(initial=AppState.SLEEPING)
    if awake:
        state.wake()
    api = GojoAPI(
        settings=_settings(tmp),
        brain=MockProvider(),
        state=state,
        transcript=TranscriptStore(tmp / "data"),
        version="test",
    )
    return api


def test_chat_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _api(Path(tmp))
        r = api.chat("hello gojo")
        assert r["ok"] is True and r["kind"] == "reply"
        assert "hello gojo" in r["reply"]
        assert api.get_status()["status"]["state"] == AppState.ACTIVE.value
        items = api.get_transcript()["items"]
        assert [i["role"] for i in items] == ["user", "model"]
    print("PASS test_chat_roundtrip")


def test_sleeping_rejects_chat() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _api(Path(tmp), awake=False)
        r = api.chat("hello")
        assert r["ok"] is False
        assert "sleeping" in r["error"]
        assert api.get_transcript()["items"] == []
    print("PASS test_sleeping_rejects_chat")


def test_sleep_command() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _api(Path(tmp))
        r = api.chat("gojo, sleep")
        assert r["ok"] is True and r["kind"] == "command"
        assert r["reply"] == SLEEP_ACK
        assert api.get_status()["status"]["state"] == AppState.SLEEPING.value
        # the ack is a REAL transcript entry, not an afterthought
        assert api.get_transcript()["items"][-1]["text"] == SLEEP_ACK
    print("PASS test_sleep_command")


def test_wake_command_while_awake_is_ack() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _api(Path(tmp))
        r = api.chat("wake up")
        assert r["ok"] is True and r["kind"] == "command"
        assert r["reply"] == WAKE_ACK
        # acks are short-circuits: they do not pollute the AI history
        assert api.chat("hello")["kind"] == "reply"
    print("PASS test_wake_command_while_awake_is_ack")


def test_power_buttons() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _api(Path(tmp))
        assert api.sleep()["status"]["state"] == AppState.SLEEPING.value
        assert api.wake()["status"]["state"] == AppState.ACTIVE.value
    print("PASS test_power_buttons")


def test_pending_sleep_applies_after_reply() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _api(Path(tmp))
        state = api._state  # white-box: simulate user pressing sleep mid-think
        state.begin_thinking()
        api.sleep()
        assert state.state is AppState.THINKING and state.pending_sleep
        r = api.chat("hello")  # finishes thinking, then sleeps
        assert r["ok"] is True
        assert api.get_status()["status"]["state"] == AppState.SLEEPING.value
    print("PASS test_pending_sleep_applies_after_reply")


def test_new_chat_resets_context() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _api(Path(tmp))
        api.chat("remember this: 42")
        assert api.new_chat()["items"] == []
        assert api.get_transcript()["items"] == []
    print("PASS test_new_chat_resets_context")


def test_missing_brain_surfaces_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state = StateManager(initial=AppState.ACTIVE)
        api = GojoAPI(
            settings=_settings(Path(tmp)),
            brain=None,
            state=state,
            transcript=TranscriptStore(Path(tmp) / "data"),
            brain_error="GEMINI_API_KEY is not set.",
        )
        r = api.chat("hello")
        assert r["ok"] is False and "GEMINI_API_KEY" in r["error"]
    print("PASS test_missing_brain_surfaces_error")


def test_clear_data_wipes_transcript() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _api(Path(tmp))
        api.chat("hello")
        assert api.clear_data()["ok"] is True
        assert api.get_transcript()["items"] == []
    print("PASS test_clear_data_wipes_transcript")


def test_headless_window_calls_fail_cleanly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _api(Path(tmp))
        assert api.minimize()["ok"] is False
        assert api.close_app()["ok"] is False
    print("PASS test_headless_window_calls_fail_cleanly")


def main() -> int:
    tests = [
        test_chat_roundtrip,
        test_sleeping_rejects_chat,
        test_sleep_command,
        test_wake_command_while_awake_is_ack,
        test_power_buttons,
        test_pending_sleep_applies_after_reply,
        test_new_chat_resets_context,
        test_missing_brain_surfaces_error,
        test_clear_data_wipes_transcript,
        test_headless_window_calls_fail_cleanly,
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

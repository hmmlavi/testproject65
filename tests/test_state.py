"""Checks for the GOJO state machine.

Run:  python -m tests.test_state
"""
from __future__ import annotations

from gojo.core.state import AppState, StateManager


def _make() -> StateManager:
    return StateManager(initial=AppState.SLEEPING)


def test_boots_sleeping() -> None:
    assert _make().state is AppState.SLEEPING
    print("PASS test_boots_sleeping")


def test_wake_and_sleep() -> None:
    sm = _make()
    assert sm.wake() is AppState.ACTIVE
    assert sm.sleep() is AppState.SLEEPING
    print("PASS test_wake_and_sleep")


def test_thinking_roundtrip() -> None:
    sm = _make()
    sm.wake()
    assert sm.begin_thinking() is AppState.THINKING
    assert sm.end_thinking() is AppState.ACTIVE
    print("PASS test_thinking_roundtrip")


def test_cannot_think_while_asleep() -> None:
    sm = _make()
    assert sm.begin_thinking() is AppState.SLEEPING  # must wake first
    print("PASS test_cannot_think_while_asleep")


def test_sleep_pending_during_thinking() -> None:
    """'Gojo, sleep' mid-response: GOJO finishes thinking, THEN sleeps."""
    sm = _make()
    sm.wake()
    sm.begin_thinking()
    sm.sleep()
    assert sm.state is AppState.THINKING  # not cut off
    assert sm.pending_sleep is True
    assert sm.end_thinking() is AppState.SLEEPING  # slept after finishing
    print("PASS test_sleep_pending_during_thinking")


def test_wake_clears_pending_sleep() -> None:
    sm = _make()
    sm.wake()
    sm.begin_thinking()
    sm.sleep()
    sm.wake()  # user presses power again
    assert sm.state is AppState.ACTIVE
    assert sm.pending_sleep is False
    print("PASS test_wake_clears_pending_sleep")


def test_listener_notified_once_per_change() -> None:
    sm = _make()
    seen: list[AppState] = []
    sm.add_listener(seen.append)
    sm.wake()
    sm.wake()  # no-op transition must not re-notify
    sm.sleep()
    assert seen == [AppState.ACTIVE, AppState.SLEEPING]
    print("PASS test_listener_notified_once_per_change")


def test_listener_crash_is_isolated() -> None:
    sm = _make()
    calls: list[AppState] = []

    def broken(_state: AppState) -> None:
        raise RuntimeError("bad listener")

    sm.add_listener(broken)
    sm.add_listener(calls.append)
    sm.wake()
    assert calls == [AppState.ACTIVE]
    print("PASS test_listener_crash_isolated")


def main() -> int:
    tests = [
        test_boots_sleeping,
        test_wake_and_sleep,
        test_thinking_roundtrip,
        test_cannot_think_while_asleep,
        test_sleep_pending_during_thinking,
        test_wake_clears_pending_sleep,
        test_listener_notified_once_per_change,
        test_listener_crash_is_isolated,
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

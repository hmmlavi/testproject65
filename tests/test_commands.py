"""Checks for direct-command detection (sleep / wake).

Run:  python -m tests.test_commands
"""
from __future__ import annotations

from gojo.core.commands import detect_direct_command

SLEEP_CASES = [
    "sleep",
    "gojo, sleep",
    "Sleep.",
    "please go to sleep",
    "so ja",
    "so jao",
    "so jaana",
    "sona",
]

WAKE_CASES = [
    "wake up",
    "wake up gojo",
    "Wake Up",
    "awake",
    "jaago",
    "jaggo",
    "utho",
]

NO_COMMAND_CASES = [
    "",
    "   ",
    "good morning",
    "i am very sleepy today",      # "sleepy" != "sleep"
    "tell me a joke",
    "acha theek hai",
    # sleep inside a long sentence must NOT trigger a command
    "kal raat main bohot der tak jaag ke raha tha aur ab mera sleep schedule bilkul bigad gaya hai boss",
]


def test_sleep_detection() -> None:
    for text in SLEEP_CASES:
        got = detect_direct_command(text)
        assert got == "sleep", f"{text!r} -> {got!r}, expected 'sleep'"
    print("PASS test_sleep_detection")


def test_wake_detection() -> None:
    for text in WAKE_CASES:
        got = detect_direct_command(text)
        assert got == "wake", f"{text!r} -> {got!r}, expected 'wake'"
    print("PASS test_wake_detection")


def test_no_false_positives() -> None:
    for text in NO_COMMAND_CASES:
        got = detect_direct_command(text)
        assert got is None, f"{text!r} -> {got!r}, expected None"
    print("PASS test_no_false_positives")


def main() -> int:
    tests = [test_sleep_detection, test_wake_detection, test_no_false_positives]
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

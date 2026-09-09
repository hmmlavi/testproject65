"""Checks for the transcript store (real transcripts, JSONL on disk).

Run:  python -m tests.test_transcript
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from gojo.transcript import TranscriptStore


def _store(tmp: Path) -> TranscriptStore:
    return TranscriptStore(tmp / "data")


def test_append_and_load_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp))
        store.append("user", "gojo, hello")
        store.append("model", "Hello boss. Kya haal hai?")
        items = store.load()
        assert len(items) == 2
        assert items[0]["role"] == "user" and items[0]["text"] == "gojo, hello"
        assert items[1]["role"] == "model"
        assert "boss" in items[1]["text"]  # unicode/Hinglish must survive
        assert "ts" in items[0]
    print("PASS test_append_and_load_roundtrip")


def test_new_session_isolates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp))
        store.append("user", "old message")
        store.new_session()
        assert store.load() == []
        store.append("user", "new message")
        items = store.load()
        assert len(items) == 1 and items[0]["text"] == "new message"
    print("PASS test_new_session_isolates")


def test_clear_all_deletes_everything() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp))
        store.append("user", "a")
        store.new_session()
        store.append("user", "b")
        store.clear_all()
        assert store.load() == []
        # store must still be usable after clear
        store.append("user", "c")
        assert len(store.load()) == 1
    print("PASS test_clear_all_deletes_everything")


def test_corrupt_line_is_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp))
        store.append("user", "good line")
        with store.current_file.open("a", encoding="utf-8") as fh:
            fh.write("this is not json\n")
        items = store.load()
        assert len(items) == 1 and items[0]["text"] == "good line"
    print("PASS test_corrupt_line_is_skipped")


def test_resume_latest_on_restart() -> None:
    """A fresh store pointed at the same data dir resumes the latest session."""
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        first = _store(data)
        first.append("user", "persisted")
        second = _store(data)  # simulates app restart
        assert [i["text"] for i in second.load()] == ["persisted"]
    print("PASS test_resume_latest_on_restart")


def main() -> int:
    tests = [
        test_append_and_load_roundtrip,
        test_new_session_isolates,
        test_clear_all_deletes_everything,
        test_corrupt_line_is_skipped,
        test_resume_latest_on_restart,
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

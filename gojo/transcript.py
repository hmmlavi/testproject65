"""Conversation transcript store — REAL transcripts, not summaries.

Phase 2 format: one session = one .jsonl file in <data>/sessions/.
Every line is one message: {"role": "user"|"model", "text": ..., "ts": ...}

Phase 6 upgrades this to SQLite + search + topic grouping. The interface
(load/append/new_session/clear_all) is what the rest of GOJO depends on,
so the backend swap stays local to this module.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class TranscriptStore:
    def __init__(self, data_dir: Path | str) -> None:
        self.dir = Path(data_dir) / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._current = self._latest_file() or self._new_file()

    # -- session management --------------------------------------------
    def _new_file(self) -> Path:
        # microseconds: fast back-to-back new sessions must not collide
        self._current = self.dir / f"session-{int(time.time() * 1_000_000)}.jsonl"
        return self._current

    def _latest_file(self) -> Path | None:
        files = sorted(self.dir.glob("session-*.jsonl"))
        return files[-1] if files else None

    @property
    def current_file(self) -> Path | None:
        return self._current

    def new_session(self) -> Path:
        """Start a fresh transcript file (previous ones are kept on disk)."""
        return self._new_file()

    def clear_all(self) -> None:
        """Delete every saved session, then start a fresh one."""
        for path in self.dir.glob("session-*.jsonl"):
            path.unlink(missing_ok=True)
        self._new_file()

    # -- messages ---------------------------------------------------------
    def append(self, role: str, text: str) -> None:
        entry = {"role": role, "text": text, "ts": int(time.time())}
        with self._current.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        """Load the current session's messages, oldest first.

        Corrupt lines are skipped, never fatal — a transcript must not be
        able to crash the app.
        """
        if self._current is None or not self._current.exists():
            return []
        out: list[dict] = []
        for line in self._current.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and "role" in entry and "text" in entry:
                out.append(entry)
        return out

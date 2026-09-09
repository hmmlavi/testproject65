"""Audio playback (Phase 3).

Windows: `winsound` from the standard library — MCI-based, plays MP3 and
WAV, no console window, no extra dependency. It's the simplest reliable
option on the target platform.

Other platforms: no built-in player — playback is reported unavailable
(development machines only; the user runs GOJO on Windows). Tests inject a
`play_fn`.

Guarantees (spec):
- temp audio files are ALWAYS deleted after playback (success or failure)
- leftovers from crashed sessions are swept at startup
- playback failures return False (the UI shows a note) — never crash
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("gojo.voice.playback")


def _default_play(path: Path) -> None:
    """Windows default: play, then block until the sound finishes."""
    import winsound

    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
    while winsound.QuerySoundPlayState() != winsound.SND_PLAY_SYNC:
        time.sleep(0.05)


class AudioPlayer:
    def __init__(self, play_fn: Callable[[Path], None] | None = None) -> None:
        self._play_fn = play_fn if play_fn is not None else (
            _default_play if sys.platform == "win32" else None
        )

    @property
    def available(self) -> bool:
        return self._play_fn is not None

    def play(self, path: Path) -> bool:
        """Play one audio file, then delete it. Returns success."""
        if not self.available:
            logger.warning("No audio player on this platform — skipping %s", path.name)
            path.unlink(missing_ok=True)
            return False
        try:
            self._play_fn(path)
            return True
        except Exception as exc:  # noqa: BLE001 — surface as False, log detail
            logger.exception("Playback failed: %s", exc)
            return False
        finally:
            path.unlink(missing_ok=True)  # no permanent audio files, ever

    @staticmethod
    def sweep_stale(audio_dir: Path, older_than_seconds: float = 3600) -> int:
        """Delete audio leftovers from previous/crashed sessions."""
        if not audio_dir.exists():
            return 0
        now = time.time()
        count = 0
        for p in audio_dir.iterdir():
            if p.is_file() and now - p.stat().st_mtime > older_than_seconds:
                p.unlink(missing_ok=True)
                count += 1
        if count:
            logger.info("swept %d stale audio files", count)
        return count

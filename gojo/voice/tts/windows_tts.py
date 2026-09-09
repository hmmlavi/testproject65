"""Windows SAPI TTS — the LOCAL fallback voice (Phase 3).

Free, offline, zero downloads: uses the voices built into Windows 10/11
(English by default; Hindi voices such as "Microsoft Hemant" are part of
the OS voice pack). Simpler quality than Fish Audio — it's the safety net,
and the router never presents it as primary when Fish is configured.

Threading note: GOJO runs TTS on a worker thread, so the SAPI COM
apartment is initialized on that thread (CoInitializeEx MTA).

`synthesize_fn` is an injectable test hook.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from .base import TTSError, TTSProvider

logger = logging.getLogger("gojo.voice.tts.windows")

MIN_WAV_BYTES = 200  # SAPI should never return anything this small for real text


class WindowsTTS(TTSProvider):
    name = "windows"

    def __init__(self, synthesize_fn=None) -> None:
        self._inject = synthesize_fn

    @property
    def available(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import pyttsx3  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def synthesize(self, text: str, out_dir: Path) -> Path:
        if not self.available:
            raise TTSError(
                "Windows TTS is not available on this platform",
                provider=self.name,
                retryable=False,
            )
        if self._inject is not None:
            return self._inject(text, out_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"gojo-windows-{int(time.time() * 1000)}-{os.urandom(2).hex()}.wav"

        # SAPI COM must be initialized on the calling (worker) thread.
        try:
            import pythoncom

            pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
        except Exception:  # noqa: BLE001 — already initialized / not needed
            pass

        try:
            import pyttsx3

            engine = pyttsx3.init()
            try:
                # Verified against the pyttsx3 source (2.71 -> 2.99): the
                # method is save_to_file(text, filename) — text FIRST — and
                # the SAPI5 driver writes a WAV via SAPI.SPFileStream.
                # (There is no saveWav/saveToWav on the Engine.)
                if not hasattr(engine, "save_to_file"):
                    raise AttributeError(
                        "installed pyttsx3 is too old for file synthesis — "
                        "fix with: pip install -U pyttsx3"
                    )
                engine.save_to_file(text, str(path))
                engine.runAndWait()
            finally:
                try:
                    engine.stop()  # no-op once the save has completed
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            path.unlink(missing_ok=True)
            raise TTSError(f"Windows TTS failed: {exc}", provider=self.name) from exc

        if not path.exists() or path.stat().st_size < MIN_WAV_BYTES:
            path.unlink(missing_ok=True)
            raise TTSError("Windows TTS produced no audio", provider=self.name)
        return path

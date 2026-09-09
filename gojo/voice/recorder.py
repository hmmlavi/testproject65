"""Microphone capture (Phase 3) — press-to-talk only.

The mic opens ONLY when the user presses Talk. No continuous listening.

Endpoint detection is a lightweight RMS silence detector (a few lines of
math — no extra model in RAM, CPU-friendly for the i5-6400).

Testable without hardware: pass a `stream_factory` that returns a fake
stream object with `.read(frames) -> (array, overflow)` and a context
manager protocol.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

from .resample import resample_to_target

logger = logging.getLogger("gojo.voice.recorder")

SAMPLE_RATE = 16000  # 16kHz mono — the standard rate for whisper models


@dataclass
class RecordingResult:
    ok: bool
    audio: object | None = None  # float32 mono ndarray @ 16kHz (numpy)
    duration: float = 0.0
    error: str | None = None
    stopped_by: str = ""  # "user" | "silence" | "max" | ""


class VoiceRecorder:
    def __init__(
        self,
        stream_factory=None,
        sample_rate: int = SAMPLE_RATE,
        silence_rms: float = 0.006,
        silence_seconds: float = 1.2,
        max_seconds: float = 60.0,
        min_seconds: float = 0.8,
        device_spec: str = "",
    ) -> None:
        self._stream_factory = stream_factory  # None -> real sounddevice
        self._device_spec = (device_spec or "").strip()  # GOJO_MIC_DEVICE
        self._last_device = ""
        self._sr = sample_rate
        self._silence_rms = silence_rms
        self._silence_seconds = silence_seconds
        self._max_seconds = max_seconds
        self._min_seconds = min_seconds

        self._stop_event = threading.Event()
        self._manual = threading.Event()
        self._thread: threading.Thread | None = None
        self._chunks: list = []
        self._fatal: str | None = None
        self._stopped_by = ""
        self._started_at = 0.0
        self._result: RecordingResult | None = None

    # -- control -------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_device(self) -> str:
        """Name+index of the device the last real capture used ('' if none
        yet or a fake stream factory was injected)."""
        return self._last_device

    def start(self) -> None:
        """Open the mic and start capturing (background thread)."""
        if self.recording:
            return
        self._stop_event.clear()
        self._manual.clear()
        self._chunks = []
        self._fatal = None
        self._stopped_by = ""
        self._result = None
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        """Ask the recorder to stop (used by the Talk button / pipeline)."""
        if self._manual.is_set():
            return
        self._manual.set()
        self._stop_event.set()

    def wait_result(self, timeout: float | None = None) -> RecordingResult:
        """Block until capture ends (auto-endpoint or manual stop) and
        return the audio. ALWAYS returns a result — never raises for
        'no mic'; errors come back in result.error."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._result is not None:
            return self._result
        # thread never produced a result (e.g. join timed out)
        return RecordingResult(ok=False, error="Recording timed out", stopped_by="max")

    # -- capture loop ----------------------------------------------------
    def _run(self) -> None:
        try:
            import numpy as np
        except Exception as exc:  # noqa: BLE001
            self._fatal = f"Audio libraries not installed (numpy missing): {exc}"
            self._finish()
            return

        try:
            if self._stream_factory is not None:
                stream = self._stream_factory()
                stream_sr = int(getattr(stream, "samplerate", 0) or self._sr)
            else:
                from . import devices

                stream, dev, stream_sr = devices.open_input_stream(self._device_spec, self._sr)
                # the stream's own reported rate wins (it is ground truth)
                stream_sr = int(getattr(stream, "samplerate", 0) or stream_sr)
                self._last_device = f"{dev['name']} (index {dev['index']}, {stream_sr} Hz)"
                logger.info("recording from: %s", self._last_device)
        except Exception as exc:  # noqa: BLE001 — no mic, no PortAudio, permissions...
            self._fatal = f"No microphone available: {exc}"
            self._finish()
            return

        try:
            with stream:
                frame = max(1, stream_sr // 4)  # ~250 ms frames at the stream rate
                silent_since: float | None = None
                while True:
                    data, _overflow = stream.read(frame)
                    chunk = data.flatten()
                    # STT (and the result contract) is 16 kHz; the stream may
                    # run at the device's native rate (e.g. 48 kHz on AudioRelay)
                    if stream_sr != self._sr:
                        chunk = resample_to_target(chunk, stream_sr, self._sr)
                    self._chunks.append(chunk)

                    elapsed = time.time() - self._started_at
                    rms = math.sqrt(sum(x * x for x in chunk) / len(chunk))
                    if rms < self._silence_rms:
                        if silent_since is None:
                            silent_since = time.time()
                    else:
                        silent_since = None

                    # classify the stop reason BEFORE breaking (the event
                    # may have been set at the same instant)
                    if self._manual.is_set():
                        self._stopped_by = "user"
                    elif elapsed >= self._max_seconds:
                        self._stopped_by = "max"
                    elif (
                        silent_since is not None
                        and elapsed >= self._min_seconds
                        and time.time() - silent_since >= self._silence_seconds
                    ):
                        self._stopped_by = "silence"
                    if self._stopped_by or self._stop_event.is_set():
                        if not self._stopped_by:
                            self._stopped_by = "user" if self._manual.is_set() else "max"
                        break
        except Exception as exc:  # noqa: BLE001 — report, never crash the app
            self._fatal = f"Microphone error: {exc}"
        self._finish()

    def _finish(self) -> None:
        try:
            import numpy as np

            if self._fatal is None and self._chunks:
                audio = np.concatenate(self._chunks)
            else:
                audio = None
            duration = 0.0
            if audio is not None:
                duration = len(audio) / self._sr
            self._result = RecordingResult(
                ok=self._fatal is None and audio is not None,
                audio=audio,
                duration=duration,
                error=self._fatal,
                stopped_by=self._stopped_by,
            )
            if self._fatal:
                logger.error("recording failed: %s", self._fatal)
            else:
                logger.info(
                    "recorded %.1fs (%s)", duration, self._stopped_by or "unknown"
                )
        except Exception as exc:  # noqa: BLE001
            self._result = RecordingResult(ok=False, error=f"Failed to collect audio: {exc}")

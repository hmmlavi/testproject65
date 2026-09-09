"""Speech-to-text (Phase 3): faster-whisper, fully local.

- Auto language detection (English / Hindi / Hinglish — no manual selection)
- CPU-only, int8 (sized for the i5-6400: `small` default, configurable)
- One model instance in RAM (loaded lazily on first use) — spec: no
  stack of heavy models
- `transcribe_fn` is an injectable test hook (tests never need the model)
"""
from __future__ import annotations

import logging

logger = logging.getLogger("gojo.voice.stt")

SIZES = ("tiny", "base", "small")


class STTError(Exception):
    """STT failed (model load, inference, ...). The pipeline turns this
    into a friendly "didn't catch that" message."""


class WhisperSTT:
    def __init__(self, model_size: str = "small", transcribe_fn=None) -> None:
        if model_size not in SIZES:
            model_size = "small"
        self._size = model_size
        self._inject = transcribe_fn
        self._model = None  # lazy: faster-whisper model

    @property
    def model_size(self) -> str:
        return self._size

    def _real_transcribe(self, audio) -> str:
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy import

            logger.info(
                "Loading whisper '%s' (CPU int8) — one-time, ~1-2 GB RAM", self._size
            )
            self._model = WhisperModel(self._size, device="cpu", compute_type="int8")
        segments, _info = self._model.transcribe(
            audio,
            language=None,  # auto-detect (EN / HI / Hinglish)
            beam_size=1,  # greedy: fast on CPU
            vad_filter=True,  # trim leading/trailing silence
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe(self, audio) -> str:
        """Audio (float32 mono @16k) -> transcript text. Raises STTError."""
        try:
            fn = self._inject if self._inject is not None else self._real_transcribe
            text = fn(audio)
        except STTError:
            raise
        except Exception as exc:  # noqa: BLE001 — model/load/inference failures
            raise STTError(f"Speech recognition failed: {exc}") from exc
        return (text or "").strip()

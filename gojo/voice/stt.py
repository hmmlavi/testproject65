"""Speech-to-text (Phase 3): faster-whisper, fully local.

- Auto language detection (English / Hindi / Hinglish — no manual selection)
- CPU-only, int8 (sized for the i5-6400: `small` default, configurable)
- One model instance in RAM (loaded lazily on first use) — spec: no
  stack of heavy models
- The FIRST call may download the model (small ~460 MB, once). The UI is
  told via `on_loading` so the user sees what is happening instead of a
  silent multi-minute wait.
- `transcribe_fn` is an injectable test hook (tests never need the model)
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("gojo.voice.stt")

SIZES = ("tiny", "base", "small")


class STTError(Exception):
    """STT failed (model load, inference, ...). The pipeline turns this
    into a friendly "didn't catch that" message."""


class WhisperSTT:
    def __init__(self, model_size: str = "small", transcribe_fn=None,
                 on_loading=None) -> None:
        if model_size not in SIZES:
            model_size = "small"
        self._size = model_size
        self._inject = transcribe_fn
        self._on_loading = on_loading  # callable(message) — announced once
        self._load_notified = False
        self._model = None  # lazy: faster-whisper model

    @property
    def model_size(self) -> str:
        return self._size

    def _notify_loading(self, message: str) -> None:
        if self._load_notified or self._on_loading is None:
            return
        self._load_notified = True
        try:
            self._on_loading(message)
        except Exception:  # noqa: BLE001 — a broken note hook must not break STT
            logger.exception("stt on_loading hook failed")

    def _load_model(self):
        from faster_whisper import WhisperModel  # lazy import

        # Announce BEFORE the load: the first run downloads the model and
        # can take minutes — the user must not stare at a frozen 'listening'.
        self._notify_loading(
            "Pehli baar: speech model load ho raha hai (pehli baar download "
            "hoti hai). Thoda intezaar karo..."
        )
        t0 = time.time()
        logger.info("loading whisper '%s' (CPU int8) — first run may download it",
                    self._size)
        self._model = WhisperModel(self._size, device="cpu", compute_type="int8")
        logger.info("whisper '%s' loaded in %.1fs", self._size, time.time() - t0)
        return self._model

    def ensure_model(self):
        """Load the model if needed (used by gojo.voice.doctor for the
        separate 'did Whisper load?' check)."""
        if self._model is None:
            self._load_model()
        return self._model

    def _real_transcribe(self, audio) -> str:
        model = self.ensure_model()
        t0 = time.time()
        segments, info = model.transcribe(
            audio,
            language=None,  # auto-detect (EN / HI / Hinglish)
            beam_size=1,  # greedy: fast on CPU
            vad_filter=True,  # trim leading/trailing silence
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        # diagnostic line: facts only — NEVER the transcript content
        logger.info(
            "stt: %.1fs audio -> %d chars (lang=%s) in %.1fs",
            len(audio) / 16000, len(text), info.language, time.time() - t0,
        )
        return text

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

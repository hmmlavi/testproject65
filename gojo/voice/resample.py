"""Sample-rate conversion for microphone capture (numpy-only, no new deps).

Whisper (and GOJO's STT pipeline) requires 16 kHz mono input. But Windows
WASAPI frequently refuses to open a stream at non-native rates — e.g. the
AudioRelay virtual microphone only accepts its native 48 kHz
(`Invalid sample rate [PaErrorCode -9997]` at 16 kHz). So streams are
opened at the DEVICE's native rate (see devices.choose_stream_rate) and
the audio is converted here before it reaches the endpoint detector, the
wake engine, or STT.

Rules:
- exact integer ratio (48000 -> 16000) : plain decimation (fast path)
- other ratios      (44100 -> 16000)   : linear interpolation
- no audio is stored or sent anywhere; input and output are float32 mono
"""
from __future__ import annotations

import numpy as np


def resample_to_target(chunk: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample float32 mono `chunk` from `from_sr` to `to_sr`.

    Same-rate input is passed through untouched (identity, no copy).
    """
    if chunk.size == 0 or from_sr == to_sr:
        return chunk
    ratio = from_sr / to_sr
    n_out = int(round(chunk.size / ratio))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    nearest = round(ratio)
    if nearest >= 1 and abs(ratio - nearest) < 1e-9:
        # 48k -> 16k is exactly 3: decimation is plenty for speech
        # (speech energy is below ~4 kHz, well inside the new Nyquist)
        return chunk[::nearest].astype(np.float32, copy=False)
    pos = np.arange(n_out, dtype=np.float64) * ratio
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, chunk.size - 1)
    i1 = np.clip(i0 + 1, 0, chunk.size - 1)
    frac = (pos - i0).astype(np.float32)
    return (chunk[i0] * (1.0 - frac) + chunk[i1] * frac).astype(np.float32)

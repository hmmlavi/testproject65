"""Audio level statistics (pure, numpy-only).

The 'is the microphone actually delivering audible audio?' measurement —
used by the Mic Test, the doctor, and the recorder's session log.
NO audio is stored, saved, or sent anywhere: in and out are just
sample counts and two numbers (peak, RMS).
"""
from __future__ import annotations

import numpy as np

# Below this PEAK, a healthy mic at normal speaking distance produces no
# detectable speech. A silent/dead virtual cable sits far below this.
SILENT_PEAK = 0.002


def level_stats(arr) -> dict:
    """{frames, peak, rms, silent} for a float32 mono buffer."""
    a = np.asarray(arr, dtype=np.float32).flatten()
    n = int(a.size)
    if n == 0:
        return {"frames": 0, "peak": 0.0, "rms": 0.0, "silent": True}
    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a * a)))
    return {
        "frames": n,
        "peak": round(peak, 5),
        "rms": round(rms, 5),
        "silent": peak < SILENT_PEAK,
    }

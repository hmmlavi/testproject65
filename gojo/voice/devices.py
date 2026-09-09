"""Microphone input-device selection (Phase 4 debugging fix).

GOJO never silently guesses which microphone to use:

- ``GOJO_MIC_DEVICE`` empty   -> the system *default* input device
- ``GOJO_MIC_DEVICE`` = index -> that exact device index (e.g. ``2``)
- ``GOJO_MIC_DEVICE`` = name  -> case-insensitive exact match, else a
  UNIQUE substring match. Ambiguous or missing names fail with a readable
  list of the devices that ARE available.

This matters for setups where the "microphone" is a virtual device —
e.g. a phone feeding Windows through AudioRelay shows up in PortAudio as
an ordinary input device (often named like ``CABLE 01 (AudioRelay)``) that
is NOT the system default.

The resolution logic is pure (device list injectable) so it is fully
testable without any audio hardware. The sounddevice layer is thin and
only runs where a real audio stack exists. Nothing here records, stores,
or sends audio — at most a peak level number comes back from the test.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("gojo.voice.devices")


class MicDeviceError(Exception):
    """Readable microphone/device problem (UI shows this, logs keep detail)."""


def _fmt(devices: list[dict]) -> str:
    return "; ".join(f"{d['name']} (index {d['index']})" for d in devices)


def _import_sd():
    try:
        import sounddevice as sd
        return sd
    except Exception as exc:  # noqa: BLE001 — no PortAudio on this machine
        raise MicDeviceError(f"sounddevice/PortAudio unavailable: {exc}") from exc


def query_input_devices() -> list[dict]:
    """Input devices PortAudio currently sees. Virtual cable mics (AudioRelay,
    VoiceMeeter, ... ) appear here as ordinary input devices. Each entry
    carries ``default_samplerate`` — the rate Windows is known to accept
    for that device (virtual mics like AudioRelay often accept ONLY this
    one)."""
    sd = _import_sd()
    devs: list[dict] = []
    for i, d in enumerate(sd.query_devices()):
        try:
            channels = int(d.get("max_input_channels") or 0)
        except (TypeError, ValueError):
            channels = 0
        if channels <= 0:
            continue
        try:
            rate = float(d.get("default_samplerate") or 0)
        except (TypeError, ValueError):
            rate = 0.0
        devs.append(
            {"index": i, "name": d["name"], "channels": channels,
             "default_samplerate": rate}
        )
    return devs


def choose_stream_rate(dev: dict, requested: int = 16000) -> int:
    """The sample rate to open the stream at.

    Uses the device's native ``default_samplerate`` when known — WASAPI
    often rejects other rates (AudioRelay: 48 kHz only; opening at 16 kHz
    fails with ``Invalid sample rate [PaErrorCode -9997]``). Captured
    audio is converted to 16 kHz afterwards (voice/resample.py). Devices
    without rate info keep the requested rate (legacy behaviour).
    Pure function — unit-testable without hardware.
    """
    try:
        rate = float(dev.get("default_samplerate") or 0)
    except (TypeError, ValueError):
        rate = 0.0
    return int(rate) if rate > 0 else int(requested)


def resolve_input_device(
    spec: str,
    devices: list[dict] | None = None,
    default_index: int | None = None,
) -> dict:
    """Resolve a ``GOJO_MIC_DEVICE`` spec to a concrete input device.

    Pure function when ``devices`` is injected (tests). Returns
    ``{"index", "name", "channels"}``; raises :class:`MicDeviceError` with
    a readable, actionable message (never an arbitrary silent fallback).
    """
    spec = (spec or "").strip()
    if devices is None:
        sd = _import_sd()
        devices = query_input_devices()
        if default_index is None:
            default_index = (sd.default.device or (-1, -1))[0]
    if not devices:
        raise MicDeviceError(
            "no input devices found — is the mic (or AudioRelay) plugged in and running?"
        )
    if spec == "":
        if default_index is None or default_index < 0:
            raise MicDeviceError(
                "system has no default input device. Set GOJO_MIC_DEVICE in .env to one of: "
                + _fmt(devices)
            )
        for d in devices:
            if d["index"] == default_index:
                return d
        raise MicDeviceError(
            f"default input index {default_index} not found; available: " + _fmt(devices)
        )
    if spec.isdigit():
        for d in devices:
            if d["index"] == int(spec):
                return d
        raise MicDeviceError(f"input device index {spec} not found; available: " + _fmt(devices))
    low = spec.lower()
    exact = [d for d in devices if d["name"].lower() == low]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise MicDeviceError(
            f"several devices share the name {spec!r}; use an index instead: " + _fmt(exact)
        )
    partial = [d for d in devices if low in d["name"].lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise MicDeviceError(
            f"device name {spec!r} matches several: " + _fmt(partial)
            + " — use the exact name or an index"
        )
    raise MicDeviceError(f"input device {spec!r} not found; available: " + _fmt(devices))


def open_input_stream(spec: str, sr: int = 16000):
    """Open a real sounddevice InputStream on the resolved device.
    Returns ``(stream, device, stream_sr)`` — the stream is opened at the
    device's native rate (see choose_stream_rate); the caller owns the
    stream and must convert audio from ``stream_sr`` to its own target."""
    dev = resolve_input_device(spec)
    sd = _import_sd()
    stream_sr = choose_stream_rate(dev, sr)
    logger.info("opening input device: %s (index %s, spec %r) at %d Hz",
                dev["name"], dev["index"], spec, stream_sr)
    stream = sd.InputStream(
        samplerate=stream_sr, channels=1, dtype="float32", device=dev["index"]
    )
    return stream, dev, stream_sr


def read_peak(spec: str, seconds: float = 1.2, sr: int = 16000) -> dict:
    """Read ~``seconds`` of audio from the resolved device and report level
    stats only. NO audio is stored, saved, or sent anywhere — the result is
    the device name, sample count, and peak amplitude."""
    import numpy as np

    stream, dev, _stream_sr = open_input_stream(spec, sr)
    frames = 0
    peak = 0.0
    n = max(1, int(sr * 0.1))
    t0 = time.time()
    try:
        with stream:
            while time.time() - t0 < seconds:
                data, _overflow = stream.read(n)
                arr = np.asarray(data).flatten()
                frames += int(arr.size)
                if arr.size:
                    p = float(np.max(np.abs(arr)))
                    if p > peak:
                        peak = p
    finally:
        logger.info("mic test done: %s — %d samples, peak %.4f", dev["name"], frames, peak)
    return {
        "ok": True,
        "device": dev["name"],
        "index": dev["index"],
        "frames": frames,
        "peak": round(peak, 4),
    }

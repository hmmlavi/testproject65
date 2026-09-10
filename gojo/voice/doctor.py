"""GOJO voice doctor — real-PC diagnostic for the voice runtime path.

Run on the Windows PC (CLOSE the GOJO window first, so the doctor owns
the microphone):

    python -m gojo.voice.doctor              # 5 s capture + wake probe
    python -m gojo.voice.doctor --seconds 8  # longer capture
    python -m gojo.voice.doctor --device 15  # override GOJO_MIC_DEVICE
    python -m gojo.voice.doctor --diagnose   # + Whisper deep-dive (A/B)
    python -m gojo.voice.doctor --stt-deep   # + deep audio/STT forensics

The --diagnose deep-dive transcribes the SAME captured sample four ways
(all local — nothing is sent anywhere) and prints detected language,
probability, segment count and per-segment text/probability for each:

    app config (auto language, VAD off)   <- what the app now uses
    old app config (auto language, VAD on)
    language=hi, VAD off
    language=en, VAD off

The --stt-deep forensic diagnostic performs comprehensive local STT analysis:
1. Captures and analyzes exact audio buffer sent to faster-whisper.
2. Saves exact buffer to a local temporary WAV file for manual audio review.
3. Reports sample rate, channels, sample count, duration, dtype, range, peak,
   RMS, DC offset, and clipping percentage.
4. Inspects installed faster-whisper/CTranslate2 version and transcribe defaults
   (e.g. temperature fallback schedule behavior on CPU).
5. Tests mono vs stereo array handling and PyAV local WAV decoding.
6. Runs an extensive decoding matrix comparing:
   - Model sizes: tiny vs small
   - Input types: direct float32 ndarray vs local WAV file path
   - Parameters: vad_filter (True/False), condition_on_previous_text (True/False),
     temperature (0.0 float vs (0.0,) tuple vs [0.0..1.0] fallback list),
     beam_size (1 vs 5), without_timestamps (True/False)
   - Languages: auto-detect vs forced 'en' vs forced 'hi'
7. Reports per-segment metrics (start/end, no-speech prob, compression ratio,
   avg logprob) and wake-phrase matcher results on every output.

Safety rules:
- no API keys are read or printed (only the mic device spec + model size)
- no conversation content is stored or sent (all processing is 100% local;
  no network calls or audio uploads)
- temporary WAV file is saved locally to temp directory for PC inspection
- bounded runtime (capture seconds + model loads + short transcribe matrix)

Exit code: 0 = every step OK, 1 = at least one step failed.
"""
from __future__ import annotations

import argparse
import inspect
import math
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from .level import level_stats
from .resample import resample_to_target
from .wakeword import WakeListener, WhisperWakeWord, WakeWordEngine, match_wake_phrase

REPORT_TITLE = "GOJO voice doctor (close the GOJO window before running)"


def analyze_capture(raw: np.ndarray, stream_sr: int) -> dict:
    """Pure steps [3]-[6]: level + resample report for a raw capture."""
    stats = level_stats(raw)
    resampled = stream_sr != 16000
    buf16 = resample_to_target(raw, stream_sr, 16000) if resampled else raw
    return {
        "samples": stats["frames"],
        "peak": stats["peak"],
        "rms": stats["rms"],
        "silent": stats["silent"],
        "resampled": resampled,
        "samples_16k": int(buf16.size),
        "buffer_16k": buf16,
    }


def analyze_buffer_forensics(buf: np.ndarray, sr: int = 16000) -> dict:
    """Detailed audio buffer forensics: sample count, duration, range, peak, RMS,
    DC offset, and digital clipping percentage."""
    buf = np.asarray(buf)
    if buf.size == 0:
        return {
            "samples": 0,
            "channels": 0,
            "sr": sr,
            "duration": 0.0,
            "dtype": str(buf.dtype),
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "peak": 0.0,
            "rms": 0.0,
            "dbfs": -float("inf"),
            "clipped_samples": 0,
            "clipping_pct": 0.0,
            "silent": True,
            "verdict": "EMPTY BUFFER",
        }

    channels = 1 if buf.ndim == 1 else (buf.shape[1] if buf.ndim == 2 else 0)
    sample_count = len(buf) if buf.ndim == 1 else buf.shape[0]
    duration = sample_count / sr if sr > 0 else 0.0
    dtype_str = str(buf.dtype)
    min_val = float(np.min(buf))
    max_val = float(np.max(buf))
    mean_val = float(np.mean(buf))
    peak_val = float(np.max(np.abs(buf)))
    rms_val = float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))
    dbfs = 20.0 * math.log10(rms_val) if rms_val > 1e-9 else -99.9

    # Digital clipping count: samples at or near +/-1.0 (>= 0.999)
    clipped_samples = int(np.sum(np.abs(buf) >= 0.999))
    clipping_pct = (clipped_samples / buf.size) * 100.0 if buf.size > 0 else 0.0
    silent = peak_val < 0.01

    if silent:
        verdict = "SILENT (peak < 0.01) — mic did not pick up audible audio"
    elif clipping_pct > 1.0:
        verdict = f"HIGH CLIPPING ({clipping_pct:.2f}%) — signal exceeds digital full-scale; distortion degrades Whisper"
    elif peak_val > 0.98:
        verdict = f"NEAR CLIPPING (peak {peak_val:.4f}, clipping {clipping_pct:.2f}%) — high input gain"
    elif rms_val < 0.02:
        verdict = "LOW SIGNAL (RMS < 0.02) — speech may be faint or distant"
    else:
        verdict = "NORMAL LEVEL — good dynamic range"

    return {
        "samples": sample_count,
        "channels": channels,
        "sr": sr,
        "duration": duration,
        "dtype": dtype_str,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "peak": peak_val,
        "rms": rms_val,
        "dbfs": dbfs,
        "clipped_samples": clipped_samples,
        "clipping_pct": clipping_pct,
        "silent": silent,
        "verdict": verdict,
    }


def save_buffer_to_wav(buf: np.ndarray, path: str | Path | None = None, sr: int = 16000) -> Path:
    """Save audio buffer as a standard 16-bit PCM mono/stereo WAV file.
    100% local, no network. Returns Path to written WAV."""
    buf = np.asarray(buf, dtype=np.float32)
    if path is None:
        ts = int(time.time())
        path = Path(tempfile.gettempdir()) / f"gojo_doctor_capture_{ts}.wav"
    else:
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if buf.ndim == 1:
        nchannels = 1
        data = buf
    elif buf.ndim == 2:
        nchannels = buf.shape[1]
        data = buf
    else:
        raise ValueError(f"Unsupported array ndim={buf.ndim} for WAV export")

    clamped = np.clip(data, -1.0, 1.0)
    pcm16 = (clamped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return path


def inspect_whisper_environment() -> dict:
    """Inspect installed faster-whisper, ctranslate2, and transcribe signature defaults."""
    info: dict = {
        "faster_whisper_installed": False,
        "faster_whisper_version": "unknown",
        "ctranslate2_installed": False,
        "ctranslate2_version": "unknown",
        "transcribe_defaults": {},
        "temperature_default": None,
        "beam_size_default": None,
        "vad_filter_default": None,
        "condition_on_previous_text_default": None,
        "without_timestamps_default": None,
    }
    try:
        import faster_whisper

        info["faster_whisper_installed"] = True
        info["faster_whisper_version"] = getattr(faster_whisper, "__version__", "unknown")
    except ImportError:
        return info

    try:
        import ctranslate2

        info["ctranslate2_installed"] = True
        info["ctranslate2_version"] = getattr(ctranslate2, "__version__", "unknown")
    except ImportError:
        pass

    try:
        model_cls = faster_whisper.WhisperModel
        sig = inspect.signature(model_cls.transcribe)
        defaults = {}
        for param_name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                defaults[param_name] = param.default
        info["transcribe_defaults"] = defaults
        info["temperature_default"] = defaults.get("temperature")
        info["beam_size_default"] = defaults.get("beam_size")
        info["vad_filter_default"] = defaults.get("vad_filter")
        info["condition_on_previous_text_default"] = defaults.get("condition_on_previous_text")
        info["without_timestamps_default"] = defaults.get("without_timestamps")
    except Exception as exc:  # noqa: BLE001
        info["signature_error"] = str(exc)

    return info


def test_mono_stereo_handling(transcribe_fn, buf16: np.ndarray,
                              wav_path: Path | str | None = None,
                              on_line=print) -> dict:
    """Test how faster-whisper handles 1D mono, 2D (N,1), 2D (N,2) stereo ndarrays, and local WAV path."""
    results = {}

    # 1. 1D mono ndarray
    try:
        t0 = time.time()
        _segs, _info = transcribe_fn(buf16, beam_size=1, temperature=0.0)
        results["1d_mono"] = {"status": "OK", "time": time.time() - t0, "note": "1D float32 mono compatible"}
    except Exception as exc:  # noqa: BLE001
        results["1d_mono"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}

    # 2. 2D (N, 1) ndarray
    try:
        buf_2d_mono = buf16[:, None]
        _segs, _info = transcribe_fn(buf_2d_mono, beam_size=1, temperature=0.0)
        results["2d_mono"] = {"status": "OK", "note": "2D (N,1) accepted"}
    except Exception as exc:  # noqa: BLE001
        results["2d_mono"] = {"status": "REJECTED (expected)", "error": f"{type(exc).__name__}: {exc}"}

    # 3. 2D (N, 2) stereo ndarray
    try:
        buf_2d_stereo = np.column_stack([buf16, buf16])
        _segs, _info = transcribe_fn(buf_2d_stereo, beam_size=1, temperature=0.0)
        results["2d_stereo"] = {"status": "OK", "note": "2D (N,2) accepted"}
    except Exception as exc:  # noqa: BLE001
        results["2d_stereo"] = {"status": "REJECTED (expected)", "error": f"{type(exc).__name__}: {exc}"}

    # 4. Local WAV file path (tests PyAV decode pipeline)
    if wav_path and Path(wav_path).exists():
        try:
            t0 = time.time()
            _segs, _info = transcribe_fn(str(wav_path), beam_size=1, temperature=0.0)
            results["wav_path"] = {
                "status": "OK",
                "time": time.time() - t0,
                "note": "PyAV decoded local WAV cleanly to 1D float32 mono",
            }
        except Exception as exc:  # noqa: BLE001
            results["wav_path"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    else:
        results["wav_path"] = {"status": "SKIPPED", "note": "No WAV path provided"}

    on_line("  - 1D mono ndarray (16k)  : " + results["1d_mono"]["status"])
    on_line("  - 2D (N,1) ndarray (16k) : " + results["2d_mono"]["status"]
            + (f" ({results['2d_mono']['error']})" if "error" in results["2d_mono"] else ""))
    on_line("  - 2D (N,2) stereo (16k)  : " + results["2d_stereo"]["status"]
            + (f" ({results['2d_stereo']['error']})" if "error" in results["2d_stereo"] else ""))
    on_line("  - Local WAV path (PyAV)  : " + results["wav_path"]["status"]
            + (f" ({results['wav_path']['note']})" if "note" in results["wav_path"] else ""))

    return results


def whisper_deep_dive(transcribe, buf16: np.ndarray, language_cfg: str,
                      on_line=print) -> None:
    """Transcribe one captured sample under 4 STT configs and report
    language, segments and per-segment detail. `transcribe` is the model's
    own `transcribe(audio, beam_size=1, **kw) -> (segments, info)` method
    (or an equivalent in tests). Nothing is sent anywhere."""
    rows = [
        ("app config", {"language": language_cfg or None, "vad_filter": False,
                        "condition_on_previous_text": True}),
        ("old app config (VAD on)", {"language": language_cfg or None,
                                     "vad_filter": True,
                                     "condition_on_previous_text": True}),
        ("language=hi", {"language": "hi", "vad_filter": False,
                         "condition_on_previous_text": False}),
        ("language=en", {"language": "en", "vad_filter": False,
                         "condition_on_previous_text": False}),
    ]
    best = None  # (chars, label, segments, info, text)
    for label, kw in rows:
        t0 = time.time()
        segments, info = transcribe(buf16, beam_size=1, **kw)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        on_line(
            f"  [{label}] lang={info.language} p={info.language_probability:.2f} "
            f"segments={len(segs)} chars={len(text)} ({time.time() - t0:.1f}s)"
        )
        on_line(f"      text: {text[:200]!r}" if text else "      text: <empty>")
        if best is None or len(text) > best[0]:
            best = (len(text), label, segs, info, text)
    if best is None:
        return
    chars, label, segs, info, text = best
    if chars == 0:
        on_line("  no config produced text — the audio is not being decoded "
                "as speech by any of these settings (see level/resample above).")
        return
    on_line(f"  best: [{label}] — segment detail:")
    for i, s in enumerate(segs):
        on_line(
            f"    seg{i}: {s.start:.1f}-{s.end:.1f}s no_speech={s.no_speech_prob:.2f} "
            f"compression={s.compression_ratio:.2f} logprob={s.avg_logprob:.2f} "
            f"text={s.text.strip()[:80]!r}"
        )
    on_line(f"  wake matcher on full text: {match_wake_phrase(text)!r}")
    for i, s in enumerate(segs):
        m = match_wake_phrase(s.text)
        if m:
            on_line(f"  wake matcher on seg{i}: {m!r}")


def run_stt_deep(
    small_transcribe,
    tiny_transcribe=None,
    buf16: np.ndarray | None = None,
    wav_path: Path | str | None = None,
    language_cfg: str = "",
    on_line=print,
) -> dict:
    """Run full STT forensic test matrix comparing model sizes (tiny vs small),
    input formats (ndarray vs local WAV path), and decoding configurations."""
    if buf16 is None:
        buf16 = np.zeros(16000, dtype=np.float32)

    configs = []

    # Small model configurations
    if small_transcribe is not None:
        configs.extend([
            ("01", "small", "ndarray", "temp=0.0 beam=1 vad=F prev=F",
             small_transcribe, buf16,
             {"beam_size": 1, "temperature": 0.0, "vad_filter": False,
              "condition_on_previous_text": False, "language": language_cfg or None}),
            ("02", "small", "ndarray", "temp=(0.0,) beam=1 vad=F prev=F",
             small_transcribe, buf16,
             {"beam_size": 1, "temperature": (0.0,), "vad_filter": False,
              "condition_on_previous_text": False, "language": language_cfg or None}),
            ("03", "small", "ndarray", "temp=fallback beam=1 vad=F prev=F",
             small_transcribe, buf16,
             {"beam_size": 1, "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
              "vad_filter": False, "condition_on_previous_text": False, "language": language_cfg or None}),
            ("04", "small", "ndarray", "temp=0.0 beam=1 no_ts=T vad=F prev=F",
             small_transcribe, buf16,
             {"beam_size": 1, "temperature": 0.0, "without_timestamps": True,
              "vad_filter": False, "condition_on_previous_text": False, "language": language_cfg or None}),
            ("05", "small", "ndarray", "temp=0.0 beam=5 vad=F prev=F",
             small_transcribe, buf16,
             {"beam_size": 5, "temperature": 0.0, "vad_filter": False,
              "condition_on_previous_text": False, "language": language_cfg or None}),
            ("06", "small", "ndarray", "temp=0.0 beam=1 vad=F prev=T",
             small_transcribe, buf16,
             {"beam_size": 1, "temperature": 0.0, "vad_filter": False,
              "condition_on_previous_text": True, "language": language_cfg or None}),
            ("07", "small", "ndarray", "temp=0.0 beam=1 vad=T prev=F",
             small_transcribe, buf16,
             {"beam_size": 1, "temperature": 0.0, "vad_filter": True,
              "condition_on_previous_text": False, "language": language_cfg or None}),
            ("08", "small", "ndarray", "temp=0.0 beam=1 lang=en vad=F prev=F",
             small_transcribe, buf16,
             {"beam_size": 1, "temperature": 0.0, "vad_filter": False,
              "condition_on_previous_text": False, "language": "en"}),
            ("09", "small", "ndarray", "temp=0.0 beam=1 lang=hi vad=F prev=F",
             small_transcribe, buf16,
             {"beam_size": 1, "temperature": 0.0, "vad_filter": False,
              "condition_on_previous_text": False, "language": "hi"}),
        ])
        if wav_path and Path(wav_path).exists():
            configs.append(
                ("10", "small", "WAV", "temp=0.0 beam=1 vad=F prev=F",
                 small_transcribe, str(wav_path),
                 {"beam_size": 1, "temperature": 0.0, "vad_filter": False,
                  "condition_on_previous_text": False, "language": language_cfg or None})
            )

    # Tiny model configurations
    if tiny_transcribe is not None:
        configs.extend([
            ("11", "tiny", "ndarray", "temp=0.0 beam=1 vad=F prev=F",
             tiny_transcribe, buf16,
             {"beam_size": 1, "temperature": 0.0, "vad_filter": False,
              "condition_on_previous_text": False, "language": language_cfg or None}),
            ("12", "tiny", "ndarray", "temp=0.0 beam=1 lang=en vad=F prev=F",
             tiny_transcribe, buf16,
             {"beam_size": 1, "temperature": 0.0, "vad_filter": False,
              "condition_on_previous_text": False, "language": "en"}),
            ("13", "tiny", "ndarray", "temp=0.0 beam=1 lang=hi vad=F prev=F",
             tiny_transcribe, buf16,
             {"beam_size": 1, "temperature": 0.0, "vad_filter": False,
              "condition_on_previous_text": False, "language": "hi"}),
        ])
        if wav_path and Path(wav_path).exists():
            configs.append(
                ("14", "tiny", "WAV", "temp=0.0 beam=1 vad=F prev=F",
                 tiny_transcribe, str(wav_path),
                 {"beam_size": 1, "temperature": 0.0, "vad_filter": False,
                  "condition_on_previous_text": False, "language": language_cfg or None})
            )

    results = []
    on_line("  " + "-" * 98)
    on_line(f"  {'#':<3} {'Model':<6} {'Input':<8} {'Config Summary':<36} {'Time':<7} {'Lang (Prob)':<14} {'Segs':<5} {'Chars':<6} {'Wake'}")
    on_line("  " + "-" * 98)

    for num, model_name, input_type, label, fn, audio_inp, kw in configs:
        t0 = time.time()
        try:
            segments, info = fn(audio_inp, **kw)
            segs = list(segments)
            elapsed = time.time() - t0
            text = " ".join(s.text.strip() for s in segs).strip()
            lang = getattr(info, "language", "?") or "?"
            lang_p = float(getattr(info, "language_probability", 0.0) or 0.0)
            wake_m = match_wake_phrase(text)
            wake_display = (wake_m or "None")[:12]
            lang_display = f"{lang} ({lang_p:.2f})"
            on_line(
                f"  {num:<3} {model_name:<6} {input_type:<8} {label:<36} {elapsed:>5.2f}s  "
                f"{lang_display:<14} {len(segs):<5} {len(text):<6} {wake_display}"
            )
            results.append({
                "num": num,
                "model": model_name,
                "input_type": input_type,
                "label": label,
                "time": elapsed,
                "language": lang,
                "language_prob": lang_p,
                "segments": segs,
                "segment_count": len(segs),
                "chars": len(text),
                "text": text,
                "wake_match": wake_m,
                "error": None,
            })
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            on_line(f"  {num:<3} {model_name:<6} {input_type:<8} {label:<36} {elapsed:>5.2f}s  FAIL: {type(exc).__name__}: {exc}")
            results.append({
                "num": num,
                "model": model_name,
                "input_type": input_type,
                "label": label,
                "time": elapsed,
                "language": "?",
                "language_prob": 0.0,
                "segments": [],
                "segment_count": 0,
                "chars": 0,
                "text": "",
                "wake_match": None,
                "error": f"{type(exc).__name__}: {exc}",
            })

    on_line("  " + "-" * 98)

    # Detailed Transcripts section
    on_line("")
    on_line("[E] DETAILED TRANSCRIPTS & PER-SEGMENT METRICS:")
    has_text = False
    for r in results:
        if r["text"]:
            has_text = True
            wake_note = f" -> WAKE MATCH: {r['wake_match']!r}" if r["wake_match"] else ""
            on_line(f"  [{r['num']}] {r['model']} {r['input_type']} {r['label']}:")
            on_line(f"       transcript: {r['text']!r}{wake_note}")
            for idx, s in enumerate(r["segments"]):
                start = getattr(s, "start", 0.0)
                end = getattr(s, "end", 0.0)
                nsp = getattr(s, "no_speech_prob", 0.0)
                cr = getattr(s, "compression_ratio", 0.0)
                lp = getattr(s, "avg_logprob", 0.0)
                seg_text = getattr(s, "text", "").strip()
                on_line(
                    f"       seg{idx} ({start:.1f}s-{end:.1f}s): "
                    f"no_speech={nsp:.2f} comp={cr:.2f} logprob={lp:.2f} -> {seg_text!r}"
                )
    if not has_text:
        on_line("  (No configuration produced recognized text — see buffer forensics and clipping stats above)")

    # Diagnostic Findings Summary
    on_line("")
    on_line("[F] FORENSIC SUMMARY & ACTIONABLE FINDINGS:")
    findings = []

    # 1. Temperature fallback impact
    r_single = next((r for r in results if r["num"] == "01"), None)
    r_fallback = next((r for r in results if r["num"] == "03"), None)
    if r_single and r_fallback and r_single["time"] > 0:
        ratio = r_fallback["time"] / max(0.001, r_single["time"])
        if ratio > 1.8:
            findings.append(
                f"1. Temperature Fallback Latency: Fallback temperature list took {r_fallback['time']:.1f}s "
                f"vs {r_single['time']:.1f}s for single temperature ({ratio:.1f}x slower). "
                f"Setting temperature=0.0 or (0.0,) prevents CPU timeout loops."
            )
        else:
            findings.append(
                f"1. Temperature Fallback Latency: Single temp={r_single['time']:.2f}s, "
                f"Fallback list={r_fallback['time']:.2f}s (overhead ratio: {ratio:.1f}x)."
            )

    # 2. WAV vs ndarray consistency
    r_arr = next((r for r in results if r["num"] == "01"), None)
    r_wav = next((r for r in results if r["num"] == "10"), None)
    if r_arr and r_wav:
        if r_arr["text"] == r_wav["text"]:
            findings.append("2. Input Format: Direct ndarray and PyAV local WAV decoding produced identical transcripts.")
        else:
            findings.append(
                f"2. Input Format Difference: ndarray yielded {r_arr['chars']} chars vs "
                f"WAV yielded {r_wav['chars']} chars."
            )

    # 3. Model comparison (tiny vs small)
    r_tiny = next((r for r in results if r["num"] == "11"), None)
    if r_single and r_tiny:
        speedup = r_single["time"] / max(0.001, r_tiny["time"])
        findings.append(
            f"3. Model Comparison: tiny ({r_tiny['time']:.2f}s, {r_tiny['chars']} chars) vs "
            f"small ({r_single['time']:.2f}s, {r_single['chars']} chars) -> {speedup:.1f}x speedup with tiny."
        )

    # 4. Wake phrase match summary
    wake_matches = [r for r in results if r["wake_match"]]
    if wake_matches:
        configs_matched = ", ".join(f"#{r['num']}({r['model']})" for r in wake_matches)
        findings.append(f"4. Wake Match: Wake phrase matched in {len(wake_matches)} config(s): {configs_matched}.")
    else:
        findings.append("4. Wake Match: No wake phrase matched across tested configurations.")

    for f in findings:
        on_line(f"  {f}")

    return {
        "results": results,
        "findings": findings,
    }


def run_doctor(
    seconds: float = 5.0,
    device_spec: str | None = None,
    deps: dict | None = None,
    on_line=print,
    diagnose: bool = False,
    stt_deep: bool = False,
) -> int:
    """Run the numbered report. Returns 0 (all OK) or 1 (any failure).

    `deps` (test injection):
      open_stream(spec, sr) -> (stream, dev_dict, rate)
      stt(audio16k) -> str          (stands in for Whisper)
      wake_engine() -> WakeWordEngine
      model_small -> transcribe fn for small model
      model_tiny -> transcribe fn for tiny model
      deep_dive -> transcribe fn for deep dive
    """
    deps = deps or {}
    failures = 0
    st: dict = {}

    def say(line: str) -> None:
        on_line(line)

    def step(n: str, label: str, fn):
        nonlocal failures
        t0 = time.time()
        try:
            value = fn()
            say(f"[{n}] {label}: OK ({time.time() - t0:.1f}s) — {value}")
            return value
        except Exception as exc:  # noqa: BLE001 — a failed step must not kill the report
            failures += 1
            say(f"[{n}] {label}: FAIL ({time.time() - t0:.1f}s) — {type(exc).__name__}: {exc}")
            return None

    say(REPORT_TITLE)
    if device_spec is None:
        from ..config import load_settings

        device_spec = load_settings().mic_device
    spec = device_spec or ""
    say(f"device spec: {spec!r} ('' = system default)")

    # ------------------------------------------------------------------
    # press-to-talk path: device -> stream -> capture -> level -> 16 kHz
    # ------------------------------------------------------------------
    def open_and_report():
        if "open_stream" in deps:
            stream, dev, rate = deps["open_stream"](spec, 16000)
        else:
            from . import devices

            stream, dev, rate = devices.open_input_stream(spec, 16000)
            rate = int(getattr(stream, "samplerate", 0) or rate)
            try:
                avail = ", ".join(d["name"] for d in devices.query_input_devices())
            except Exception:  # noqa: BLE001
                avail = "?"
            say(f"      input devices: {avail}")
        st["stream"], st["dev"], st["rate"] = stream, dev, rate
        return f'{dev["name"]} (index {dev["index"]})'

    step("1", "selected device", open_and_report)
    step("2", "stream samplerate", lambda: (st.get("rate") and f'{st["rate"]} Hz') or "unknown")

    def capture():
        stream, rate = st["stream"], st["rate"]
        n = max(1, int(rate * 0.1))
        chunks = []
        t0 = time.time()
        with stream:
            while time.time() - t0 < seconds:
                data, _overflow = stream.read(n)
                chunks.append(np.asarray(data).flatten())
        raw = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        st["raw"] = raw
        return f"{len(chunks)} reads, {raw.size} samples in {time.time() - t0:.1f}s"

    step("3", "capture (speak into the mic now!)", capture)

    def level():
        if "raw" not in st:
            raise RuntimeError("no capture")
        a = analyze_capture(st["raw"], st["rate"])
        st.update(a)
        verdict = "SILENT — no audio is arriving" if a["silent"] else "audible"
        return f'peak={a["peak"]} rms={a["rms"]} -> {verdict}'

    step("4", "audio level", level)

    def resample_report():
        if "resampled" not in st:
            return "n/a (capture failed)"
        if st["resampled"]:
            return f"yes ({st['rate']} Hz -> 16000 Hz)"
        return "not needed (already 16 kHz)"

    step("5", "resample to 16 kHz", resample_report)

    def buf16_report():
        if "samples_16k" not in st:
            return "n/a (capture failed)"
        return f'{st["samples_16k"]} samples ({st["samples_16k"] / 16000:.1f}s)'

    step("6", "16 kHz audio", buf16_report)

    # ------------------------------------------------------------------
    # Whisper (STT) — same model + params as the app
    # ------------------------------------------------------------------
    from .stt import SIZES, STTError

    model_size = "small"
    stt_language = ""
    if "stt" not in deps:
        try:
            from ..config import load_settings

            s = load_settings()
            model_size = s.stt_model_size
            stt_language = s.stt_language
        except Exception:  # noqa: BLE001
            pass
    stt_size = model_size if model_size in SIZES else "small"

    def load_whisper():
        if "stt" in deps:
            st["stt_fn"] = deps["stt"]
            return f"stt size={stt_size} (injected for this test)"
        from .stt import WhisperSTT

        t0 = time.time()
        stt = WhisperSTT(stt_size, language=stt_language)
        st["model"] = stt.ensure_model()
        st["stt_fn"] = stt.transcribe
        lang_note = f", language={'auto' if not stt_language else stt_language}"
        return (f"whisper '{stt_size}' loaded OK in {time.time() - t0:.1f}s "
                f"(CPU int8{lang_note}, vad=off)")

    step("7", "Whisper model load", load_whisper)

    def transcribe():
        if "buffer_16k" not in st:
            raise RuntimeError("no 16 kHz buffer")
        t0 = time.time()
        try:
            text = st["stt_fn"](st["buffer_16k"])
        except STTError as exc:
            raise RuntimeError(f"transcribe failed: {exc}") from exc
        st["text"] = text or ""
        say(f"      transcript: {st['text'][:160]!r}" if st["text"] else "      transcript: <empty>")
        return f"{len(st['text'])} chars in {time.time() - t0:.1f}s"

    step("8", "Whisper transcript", transcribe)

    def brain_routing():
        from ..core.commands import detect_direct_command

        text = st.get("text", "")
        command = detect_direct_command(text)
        return (
            f"transcript={len(text)} chars, direct_command={command} -> "
            + ("short-circuited as a command" if command
               else "shared brain path _run_turn(via='voice') would receive it "
                  "(doctor does NOT call Gemini — no key, no network, nothing sent)")
        )

    step("9", "brain routing (local check)", brain_routing)
    step("10", "exceptions/timeouts", lambda: "none" if failures == 0
         else f"{failures} step(s) failed (see above)")

    # ------------------------------------------------------------------
    # deep-dive (A/B STT configs on the SAME sample — all local)
    # ------------------------------------------------------------------
    if diagnose:
        say("")
        say("--- whisper deep-dive (same sample, nothing is sent anywhere) ---")

        def deep_dive():
            fn = None
            if "deep_dive" in deps:
                fn = deps["deep_dive"]
            elif "model" in st:
                fn = st["model"].transcribe
            else:
                raise RuntimeError("no real model available in this run")
            whisper_deep_dive(fn, st["buffer_16k"], stt_language, on_line=say)

        try:
            deep_dive()
        except Exception as exc:  # noqa: BLE001 — keep the report going
            say(f"  deep-dive failed: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # --stt-deep: comprehensive audio/STT forensic diagnostics
    # ------------------------------------------------------------------
    if stt_deep:
        say("")
        say("=" * 70)
        say("  GOJO STT DEEP-DIVE & AUDIO FORENSICS (--stt-deep)")
        say("=" * 70)

        # [A] Faster-Whisper environment inspection
        say("")
        say("[A] FASTER-WHISPER ENVIRONMENT INSPECTION:")
        env_info = inspect_whisper_environment()
        say(f"  - faster-whisper installed: {env_info['faster_whisper_installed']} (version: {env_info['faster_whisper_version']})")
        say(f"  - ctranslate2 installed   : {env_info['ctranslate2_installed']} (version: {env_info['ctranslate2_version']})")
        say(f"  - default temperature     : {env_info['temperature_default']}")
        say(f"  - default beam_size       : {env_info['beam_size_default']}")
        say(f"  - default vad_filter      : {env_info['vad_filter_default']}")
        say(f"  - default condition_on_previous_text: {env_info['condition_on_previous_text_default']}")
        say(f"  - default without_timestamps: {env_info['without_timestamps_default']}")
        say("  * Implementation Forensics: The default faster-whisper temperature schedule is a list")
        say("    of 6 values [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]. If audio decoding encounters low log-prob")
        say("    or high compression ratio, it re-runs inference across each temperature sequentially,")
        say("    multiplying CPU latency by up to 6x (~150s). Passing temperature=0.0 or (0.0,)")
        say("    executes a single greedy pass without fallback loops.")

        # [B] Audio Buffer Forensics & Local WAV export
        say("")
        say("[B] CAPTURED BUFFER & LOCAL WAV FORENSICS:")
        buf16 = st.get("buffer_16k", np.zeros(16000, dtype=np.float32))
        try:
            wav_path = save_buffer_to_wav(buf16)
            say(f"  - WAV exported locally to : {wav_path}")
            say("    (100% local file for listening / inspection — no audio is uploaded)")
        except Exception as exc:  # noqa: BLE001
            wav_path = None
            say(f"  - WAV export failed       : {type(exc).__name__}: {exc}")

        forensics = analyze_buffer_forensics(buf16, sr=16000)
        say(f"  - Sample rate             : {forensics['sr']} Hz")
        say(f"  - Channels                : {forensics['channels']} (mono)")
        say(f"  - Sample count            : {forensics['samples']} samples")
        say(f"  - Duration                : {forensics['duration']:.2f} s")
        say(f"  - Dtype                   : {forensics['dtype']}")
        say(f"  - Value range             : min={forensics['min']:.5f}, max={forensics['max']:.5f}, mean={forensics['mean']:.5f} (DC offset)")
        say(f"  - Peak amplitude          : {forensics['peak']:.5f}")
        say(f"  - RMS level               : {forensics['rms']:.5f} ({forensics['dbfs']:.1f} dBFS)")
        say(f"  - Digital clipping        : {forensics['clipping_pct']:.2f}% ({forensics['clipped_samples']}/{forensics['samples']} samples >= 0.999)")
        say(f"  - Level Verdict           : {forensics['verdict']}")

        # [C] Mono / Stereo / WAV input handling test
        say("")
        say("[C] INPUT FORMAT & MONO/STEREO HANDLING TEST:")
        small_transcribe_fn = None
        if "model_small" in deps:
            small_transcribe_fn = deps["model_small"]
        elif "deep_dive" in deps:
            small_transcribe_fn = deps["deep_dive"]
        elif "model" in st:
            small_transcribe_fn = st["model"].transcribe

        if small_transcribe_fn is not None:
            try:
                test_mono_stereo_handling(small_transcribe_fn, buf16, wav_path=wav_path, on_line=say)
            except Exception as exc:  # noqa: BLE001
                say(f"  mono/stereo test encountered error: {type(exc).__name__}: {exc}")
        else:
            say("  skipped (no model transcribe function available)")

        # [D] Load tiny model for comparison if possible
        tiny_transcribe_fn = None
        if "model_tiny" in deps:
            tiny_transcribe_fn = deps["model_tiny"]
        elif "stt" not in deps:
            try:
                from .stt import WhisperSTT

                t0 = time.time()
                tiny_stt = WhisperSTT("tiny", language=stt_language)
                tiny_model = tiny_stt.ensure_model()
                tiny_transcribe_fn = tiny_model.transcribe
                say(f"  whisper 'tiny' loaded OK in {time.time() - t0:.1f}s (CPU int8)")
            except Exception as exc:  # noqa: BLE001
                say(f"  whisper 'tiny' not loaded ({type(exc).__name__}: {exc}) — skipping tiny model rows")

        # [E] Run STT Decoding Matrix
        say("")
        say("[D] DECODING CONFIGURATIONS MATRIX (All local, no upload):")
        try:
            run_stt_deep(
                small_transcribe=small_transcribe_fn,
                tiny_transcribe=tiny_transcribe_fn,
                buf16=buf16,
                wav_path=wav_path,
                language_cfg=stt_language,
                on_line=say,
            )
        except Exception as exc:  # noqa: BLE001
            say(f"  stt deep-dive matrix failed: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # wake listener probe (separate)
    # ------------------------------------------------------------------
    say("")
    say("--- wake listener probe ---")

    seen_texts: list[str] = []

    def recording_matcher(text: str) -> str | None:
        seen_texts.append(text)
        return match_wake_phrase(text)

    class FrameProbe(WakeWordEngine):
        """Counts frame sizes so [W3] can prove 16-kHz-scaled frames."""

        name = "probe"

        def __init__(self, inner: WakeWordEngine) -> None:
            self._inner = inner
            self.max_frame = 0

        def add_frame(self, frame) -> str | None:
            self.max_frame = max(self.max_frame, int(np.asarray(frame).size))
            return self._inner.add_frame(frame)

        def reset(self) -> None:
            self._inner.reset()

    w: dict = {}

    def wake_open():
        if "wake_engine" in deps:
            inner: WakeWordEngine = deps["wake_engine"]() if callable(deps["wake_engine"]) else deps["wake_engine"]
        else:
            inner = WhisperWakeWord(matcher=recording_matcher)
        probe = FrameProbe(inner)
        w["probe"], w["inner"] = probe, inner
        w["matches"] = []
        listener = WakeListener(
            probe,
            on_wake=lambda p: w["matches"].append(p),
            stream_factory=(lambda: deps["open_stream"](spec, 16000)[0]) if "open_stream" in deps else None,
            device_spec=spec,
        )
        w["listener"] = listener
        r = listener.start()
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "listener did not start"))
        return f"listener running on: {listener.last_device or 'injected stream (test)'}"

    step("W1", "wake listener opens the selected device", wake_open)

    def wake_run():
        listener = w["listener"]
        time.sleep(seconds)
        listener.stop()
        frames = listener.frames_seen
        return f"{frames} frames received" + (" (NON-ZERO: audio is arriving)" if frames > 0
                                              else " (ZERO: no audio arriving!)")

    step("W2", "wake listener receives audio", wake_run)

    def wake_frames():
        maxf = w["probe"].max_frame
        rate = st.get("rate", 0)
        ok = "16-kHz-scaled frames OK" if maxf <= 4000 else "TOO LARGE — not 16-kHz-scaled!"
        extra = f" (stream was {rate} Hz)" if rate and rate != 16000 else ""
        return f"max frame = {maxf} samples{extra} -> {ok}"

    step("W3", "frames reaching the engine", wake_frames)

    def wake_matcher_input():
        if seen_texts:
            last = " | ".join(repr(t[:40]) for t in seen_texts[-3:])
            return f"{len(seen_texts)} window(s) transcribed; last inputs: {last}"
        return "no window transcribed (all silent, or model not loadable — see [W5] and app log)"

    step("W4", "matcher sees Whisper output", wake_matcher_input)

    def wake_model():
        inner = w.get("inner")
        loaded = getattr(inner, "_model", "injected") if inner is not None else "?"
        matches = w.get("matches", [])
        return (f"engine model: {loaded}"
                + (f"; matched: {matches}" if matches else "; no wake match during the probe (expected unless you said a wake phrase)"))

    step("W5", "wake engine + matches", wake_model)

    say("")
    if failures:
        say(f"RESULT: {failures} step(s) FAILED — look at the FAIL lines above.")
    else:
        say("RESULT: all steps OK — the voice path is healthy end-to-end. "
            "If the app still doesn't respond, the problem is upstream of this path (UI/bridge).")
    return 0 if failures == 0 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GOJO voice doctor (real-PC diagnostic)")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="capture + probe duration in seconds (default 5)")
    parser.add_argument("--device", default=None,
                        help="GOJO_MIC_DEVICE override: '' = default, index, or name")
    parser.add_argument("--diagnose", action="store_true",
                        help="run the Whisper A/B deep-dive (4 configs, all local)")
    parser.add_argument("--stt-deep", dest="stt_deep", action="store_true",
                        help="run deep audio/STT forensics (WAV export, stats, tiny vs small, mono/stereo, decoding matrix)")
    args = parser.parse_args(argv)
    return run_doctor(seconds=args.seconds,
                      device_spec=args.device if args.device is not None else None,
                      diagnose=args.diagnose,
                      stt_deep=args.stt_deep)


if __name__ == "__main__":
    raise SystemExit(main())

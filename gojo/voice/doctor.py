"""GOJO voice doctor — real-PC diagnostic for the voice runtime path.

Run on the Windows PC (CLOSE the GOJO window first, so the doctor owns
the microphone):

    python -m gojo.voice.doctor              # 5 s capture + wake probe
    python -m gojo.voice.doctor --seconds 8  # longer capture
    python -m gojo.voice.doctor --device 15  # override GOJO_MIC_DEVICE
    python -m gojo.voice.doctor --diagnose   # + Whisper deep-dive (A/B)

The --diagnose deep-dive transcribes the SAME captured sample four ways
(all local — nothing is sent anywhere) and prints detected language,
probability, segment count and per-segment text/probability for each:

    app config (auto language, VAD off)   <- what the app now uses
    old app config (auto language, VAD on)
    language=hi, VAD off
    language=en, VAD off

plus, for the best result: per-segment detail (start/end, no-speech
probability, compression ratio, logprob) and what the WAKE MATCHER
returns for that actual recognized text (the real test of Hey/Hi/Hello/
Wake up/Yo Gojo against what Whisper really heard).

It walks the EXACT path the app uses, step by step, and prints a numbered
report:

    [1] selected device name/index        [6] resulting 16 kHz samples
    [2] stream samplerate                 [7] Whisper loaded (time/cache)
    [3] samples recorded                  [8] Whisper transcript (chars)
    [4] peak / RMS (audible or silent)    [9] brain routing decision
    [5] resample applied?                 [10] exceptions/timeouts
    [W1]-[W5] wake listener probe (device, frames, 16 kHz frames,
             what the matcher saw, matches)

Safety rules:
- no API keys are read or printed (only the mic device spec + model size)
- no conversation content is stored or sent (the transcript of YOUR test
  sentence appears in YOUR terminal, for YOUR diagnosis — nothing is
  uploaded anywhere; the brain step only computes the local command
  routing and never calls Gemini)
- no audio is written to disk
- bounded runtime (capture seconds + model loads + one short transcribe)

Exit code: 0 = every step OK, 1 = at least one step failed.
"""
from __future__ import annotations

import argparse
import time

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


def whisper_deep_dive(transcribe, buf16: np.ndarray, language_cfg: str,
                      on_line=print) -> None:
    """Transcribe one captured sample under 4 STT configs and report
    language, segments and per-segment detail. `transcribe` is the model's
    own `transcribe(audio, beam_size=1, **kw) -> (segments, info)` method
    (or an equivalent in tests). Nothing is sent anywhere."""
    from .wakeword import match_wake_phrase

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


def run_doctor(
    seconds: float = 5.0,
    device_spec: str | None = None,
    deps: dict | None = None,
    on_line=print,
    diagnose: bool = False,
) -> int:
    """Run the numbered report. Returns 0 (all OK) or 1 (any failure).

    `deps` (test injection):
      open_stream(spec, sr) -> (stream, dev_dict, rate)
      stt(audio16k) -> str          (stands in for Whisper)
      wake_engine() -> WakeWordEngine
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
    args = parser.parse_args(argv)
    return run_doctor(seconds=args.seconds,
                      device_spec=args.device if args.device is not None else None,
                      diagnose=args.diagnose)


if __name__ == "__main__":
    raise SystemExit(main())

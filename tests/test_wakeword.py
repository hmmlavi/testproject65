"""Phase 4 checks: wake-word detection, listener lifecycle, always-
listening control, and the wake -> command -> reply -> sleep flow.

Everything runs with labeled fakes (frame-level engine stand-in, fake
mic stream, scripted recorder, mock brain) — EXCEPT one optional
real-model test that runs where faster-whisper can load the tiny model
(the user's Windows PC) and skips honestly elsewhere.

NO test here claims microphone hardware works — that requires the user's
real PC. Run:  python -m tests.test_wakeword
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from gojo.ai.mock import MockProvider
from gojo.config import Settings
from gojo.core.commands import detect_direct_command
from gojo.core.state import AppState, StateManager
from gojo.prefs import Prefs
from gojo.transcript import TranscriptStore
from gojo.ui.api import GojoAPI
from gojo.voice.pipeline import VoicePipeline
from gojo.voice.playback import AudioPlayer
from gojo.voice.recorder import RecordingResult, VoiceRecorder
from gojo.voice.resample import resample_to_target
from gojo.voice.stt import WhisperSTT
from gojo.voice.tts.base import TTSProvider
from gojo.voice.tts.router import TTSRouter
from gojo.voice.wakeword import (
    WAKE_PHRASES,
    WhisperWakeWord,
    WakeListener,
    build_wakeword_engine,
    match_wake_phrase,
)


# ---------------------------------------------------------------------
# fakes (labeled stand-ins — no mic, no network, no model)
# ---------------------------------------------------------------------


class FakeWakeEngine:
    """Frame-counting stand-in: wakes after N frames, one-shot."""

    name = "fake"

    def __init__(self, trigger_after: int | None = None, phrase: str = "hey gojo") -> None:
        self.trigger_after = trigger_after
        self.phrase = phrase
        self.frames = 0
        self.matches = 0

    def add_frame(self, frame) -> str | None:
        self.frames += 1
        if self.trigger_after is not None and self.frames >= self.trigger_after:
            self.trigger_after = None
            self.matches += 1
            return self.phrase
        return None

    def reset(self) -> None:
        # mimics the real engine: buffer cleared; one-shot stays spent
        pass


class FakeStream:
    """Endless 16 kHz silence. `read` is what sounddevice.InputStream has."""

    def __init__(self, sr: int = 16000) -> None:
        self.sr = sr

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self, n: int):
        time.sleep(0.01)
        return np.zeros(n, dtype=np.float32), False


class FakeStreamFactory:
    def __init__(self) -> None:
        self.created = 0
        self.fail = False

    def __call__(self) -> FakeStream:
        if self.fail:
            raise RuntimeError("default input device not found")
        self.created += 1
        return FakeStream()


class FileTTS(TTSProvider):
    def __init__(self, name: str = "fish") -> None:
        self.name = name
        self.calls = 0

    def synthesize(self, text: str, out_dir: Path) -> Path:
        self.calls += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"fake-{self.calls}.mp3"
        p.write_bytes(b"ID3-fake")
        return p


class ScriptedRecorder:
    """No threads: returns a fixed (good) recording after a short busy window."""

    last_device = ""  # same attribute the real VoiceRecorder exposes

    def __init__(self, ok: bool = True, error: str | None = None) -> None:
        self._ok = ok
        self._error = error
        self.started = 0

    def start(self) -> None:
        self.started += 1

    def request_stop(self) -> None:
        pass

    def wait_result(self, timeout: float | None = None):
        time.sleep(0.2)
        if not self._ok:
            return RecordingResult(ok=False, audio=None, duration=0.0, error=self._error)
        return RecordingResult(
            ok=True, audio=np.zeros(16000, dtype=np.float32), duration=0.5, stopped_by="silence"
        )


def _build(tmp: Path, *, trigger_after: int | None = None, stt_text: str = "gojo, hello",
           autoplay: bool = True, recorder=None):
    settings = Settings(None, "mock-model", "mock", tmp, "INFO")
    state = StateManager(initial=AppState.SLEEPING)
    transcript = TranscriptStore(tmp)
    prefs = Prefs(tmp / "gojo_prefs.json", defaults={"tts_autoplay": autoplay})
    api = GojoAPI(settings=settings, brain=MockProvider(), state=state,
                  transcript=transcript, version="t", prefs=prefs)
    notes: list[tuple[str, str]] = []
    acks: list[int] = []
    played: list[Path] = []
    tts = FileTTS()
    factory = FakeStreamFactory()
    engine = FakeWakeEngine(trigger_after=trigger_after)
    pipeline = VoicePipeline(
        state=state,
        recorder=recorder or ScriptedRecorder(),
        stt=WhisperSTT(transcribe_fn=lambda a: stt_text),
        tts=TTSRouter([tts], order="fish"),
        player=AudioPlayer(play_fn=lambda p: played.append(p)),
        on_text=api._run_turn,
        prefs=prefs,
        on_note=lambda k, m: notes.append((k, m)),
        audio_dir=tmp / "audio",
        wake_engine=engine,
        wake_stream_factory=factory,
        ack_fn=lambda: acks.append(1),
    )
    api.attach_voice(pipeline)
    return {
        "api": api, "state": state, "transcript": transcript, "prefs": prefs,
        "notes": notes, "acks": acks, "played": played, "tts": tts,
        "factory": factory, "engine": engine, "pipeline": pipeline,
    }


def _wait_settled(pipeline, seconds: float = 20.0) -> None:
    deadline = time.time() + seconds
    while pipeline.busy and time.time() < deadline:
        time.sleep(0.02)
    assert not pipeline.busy, "turn did not finish"


# ---------------------------------------------------------------------
# 1. phrase matching (pure logic — the heart of detection)
# ---------------------------------------------------------------------


def test_matcher_required_phrases() -> None:
    assert match_wake_phrase("Hey Gojo") == "hey gojo"
    assert match_wake_phrase("Hi Gojo") == "hi gojo"
    assert match_wake_phrase("Hello Gojo") == "hello gojo"
    assert match_wake_phrase("Wake up Gojo") == "wake up gojo"
    assert match_wake_phrase("Yo Gojo") == "yo gojo"
    assert set(WAKE_PHRASES) == {
        "hey gojo", "hi gojo", "hello gojo", "wake up gojo", "yo gojo",
    }
    print("PASS test_matcher_required_phrases")


def test_matcher_asr_variants() -> None:
    variants = {
        "hey, gojo.": "hey gojo",
        "HII gojo": "hi gojo",
        "hello! GOJO": "hello gojo",
        "hey go jo": "hey gojo",          # name split across tokens
        "hi goh jo": "hi gojo",           # near-miss tokens
        "hey goyo": "hey gojo",           # edit-distance-1 (j/y confusion)
        "hey gogo": "hey gojo",           # edit-distance-1
        "heygojo": "hey gojo",            # fused without space
        "gojo": "hey gojo",               # bare name still wakes (strong signal)
        "yo, gojo": "yo gojo",            # punctuation
        "wake up, GOJO": "wake up gojo",  # classic variant
        "wake gojo": "wake up gojo",      # lenient "wake" prefix
        "  Hey   Gojo  ": "hey gojo",     # whitespace noise
    }
    for text, want in variants.items():
        got = match_wake_phrase(text)
        assert got == want, f"{text!r} -> {got!r}, want {want!r}"
    print("PASS test_matcher_asr_variants")


def test_matcher_rejects_invalid_input() -> None:
    for text in (
        "", "   ", "hello world", "good job boss", "i went to the store",
        "please sleep", "नमस्ते गो", "song lyrics a b c",
    ):
        assert match_wake_phrase(text) is None, f"{text!r} must not match"
    print("PASS test_matcher_rejects_invalid_input")


# ---------------------------------------------------------------------
# 2. engine: windowing, silence gating, errors, cooldown
# ---------------------------------------------------------------------


def _speaky(n: int) -> np.ndarray:
    return np.sin(np.linspace(0, 40 * np.pi, n)).astype(np.float32) * 0.05  # rms ~0.035


def test_engine_windowing_and_match() -> None:
    calls: list[str] = []

    def fake(chunk) -> str:
        calls.append("hey gojo")
        return "hey gojo"

    eng = WhisperWakeWord(
        transcribe_fn=fake, window_seconds=2.0, step_seconds=1.0,
        silence_rms=0.006, cooldown_seconds=0.05,
    )
    frame = _speaky(4000)  # 250 ms of "speech"
    out = None
    for _ in range(7):  # 1.75 s — window not full yet
        out = eng.add_frame(frame)
    assert out is None and calls == [], "must not transcribe before a full window"
    out = eng.add_frame(frame)  # 2.0 s — first full window
    assert out == "hey gojo" and len(calls) == 1
    # cooldown: more speech right after must not re-fire
    for _ in range(8):
        assert eng.add_frame(frame) is None
    assert len(calls) == 1, "cooldown must suppress re-transcription"
    time.sleep(0.06)  # cooldown elapsed — refill the window
    out = None
    for _ in range(8):
        out = eng.add_frame(frame)
        if out is not None:
            break
    assert out == "hey gojo" and len(calls) == 2
    print("PASS test_engine_windowing_and_match")


def test_engine_silence_never_transcribes() -> None:
    calls = []
    eng = WhisperWakeWord(transcribe_fn=lambda c: calls.append(1) or "hey gojo")
    silence = np.zeros(4000, dtype=np.float32)
    for _ in range(20):  # 5 s of silence
        assert eng.add_frame(silence) is None
    assert calls == [], "silent windows must skip ASR (CPU saver + privacy)"
    print("PASS test_engine_silence_never_transcribes")


def test_engine_tolerates_transcribe_errors() -> None:
    def boom(chunk):
        raise RuntimeError("model exploded")

    eng = WhisperWakeWord(transcribe_fn=boom)
    frame = _speaky(4000)
    for _ in range(9):
        assert eng.add_frame(frame) is None  # never raises, never matches
    print("PASS test_engine_tolerates_transcribe_errors")


def test_engine_model_size_is_capped() -> None:
    eng = WhisperWakeWord(model_size="large-v3")  # must clamp, never load big
    assert eng._model_size == "tiny"
    assert build_wakeword_engine().name == "whisper"
    print("PASS test_engine_model_size_is_capped")


# ---------------------------------------------------------------------
# 2b. input device selection (pure logic — no audio hardware needed)
# ---------------------------------------------------------------------


def test_resolve_input_device_pure() -> None:
    from gojo.voice.devices import MicDeviceError, resolve_input_device

    devs = [
        {"index": 0, "name": "Microphone (Realtek(R) Audio)", "channels": 2},
        {"index": 2, "name": "CABLE 01 (AudioRelay)", "channels": 2},
    ]
    # empty spec -> system default
    assert resolve_input_device("", devs, default_index=2)["name"] == "CABLE 01 (AudioRelay)"
    # no default -> actionable error listing available devices
    try:
        resolve_input_device("", devs, default_index=-1)
        raise AssertionError("must fail without a default")
    except MicDeviceError as exc:
        assert "GOJO_MIC_DEVICE" in str(exc) and "AudioRelay" in str(exc)
    # explicit index
    assert resolve_input_device("2", devs)["index"] == 2
    try:
        resolve_input_device("9", devs)
        raise AssertionError("bad index must fail")
    except MicDeviceError as exc:
        assert "available" in str(exc)
    # exact name, case-insensitive
    assert resolve_input_device("cable 01 (audiorelay)", devs)["index"] == 2
    # unique substring
    assert resolve_input_device("audiorelay", devs)["index"] == 2
    # ambiguous substring -> refuses to guess
    devs2 = devs + [{"index": 6, "name": "AudioRelay OUT", "channels": 1}]
    try:
        resolve_input_device("audiorelay", devs2)
        raise AssertionError("ambiguous name must fail")
    except MicDeviceError as exc:
        assert "matches several" in str(exc)
    # not found -> lists what IS available
    try:
        resolve_input_device("does-not-exist", devs)
        raise AssertionError("unknown name must fail")
    except MicDeviceError as exc:
        assert "not found" in str(exc) and "AudioRelay" in str(exc)
    # no devices at all
    try:
        resolve_input_device("", [], default_index=0)
        raise AssertionError("must fail with zero devices")
    except MicDeviceError as exc:
        assert "no input devices" in str(exc)
    print("PASS test_resolve_input_device_pure")


# ---------------------------------------------------------------------
# 2c. sample-rate handling (AudioRelay virtual mic = 48 kHz only)
# ---------------------------------------------------------------------


def _dominant_freq(audio: np.ndarray, sr: int = 16000) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1 / sr)
    return float(freqs[int(np.argmax(spec))])


def test_resample_48k_to_16k() -> None:
    """The AudioRelay case: 48000 -> 16000 is an exact 3:1 decimation and
    a 400 Hz tone must stay a 400 Hz tone."""
    sr = 48000
    tone = (np.sin(2 * np.pi * 400 * np.arange(sr // 4) / sr) * 0.1).astype(np.float32)
    out = resample_to_target(tone, sr, 16000)
    assert out.dtype == np.float32
    assert out.size == 4000  # 250 ms @ 16 kHz
    assert abs(_dominant_freq(out) - 400) < 5
    # same-rate input passes through untouched
    assert resample_to_target(tone, 16000, 16000) is tone
    assert resample_to_target(np.zeros(0, dtype=np.float32), 48000, 16000).size == 0
    print("PASS test_resample_48k_to_16k")


def test_resample_44100_to_16k() -> None:
    """Non-integer ratio (2.75625) goes through the interpolation path."""
    sr = 44100
    tone = (np.sin(2 * np.pi * 400 * np.arange(sr // 4) / sr) * 0.1).astype(np.float32)
    out = resample_to_target(tone, sr, 16000)
    assert out.size == 4000
    assert abs(_dominant_freq(out) - 400) < 5
    print("PASS test_resample_44100_to_16k")


def test_choose_stream_rate() -> None:
    from gojo.voice.devices import choose_stream_rate, resolve_input_device

    # the user's real device: Virtual Mic (AudioRelay), index 15, 48 kHz
    dev15 = {"index": 15, "name": "Virtual Mic (Virtual Mic for AudioRelay)",
             "channels": 2, "default_samplerate": 48000.0}
    assert choose_stream_rate(dev15) == 48000
    # device without rate info -> keep the requested rate (legacy behaviour)
    assert choose_stream_rate({"index": 0, "name": "Mic", "channels": 2}) == 16000
    assert choose_stream_rate({"default_samplerate": 0.0}, 22050) == 22050
    # resolver passes the rate through for explicit index/name selection
    devs = [
        {"index": 0, "name": "Microphone (Realtek(R) Audio)", "channels": 2,
         "default_samplerate": 44100.0},
        dev15,
    ]
    assert resolve_input_device("15", devs)["default_samplerate"] == 48000.0
    assert resolve_input_device("audiorelay", devs)["default_samplerate"] == 48000.0
    assert choose_stream_rate(resolve_input_device("", devs, default_index=0)) == 44100
    print("PASS test_choose_stream_rate")


class Fake48kStream:
    """Mimics the AudioRelay virtual mic: declares samplerate=48000 (like a
    real sounddevice InputStream) and serves endless 400 Hz tone frames."""

    samplerate = 48000

    def __init__(self) -> None:
        self._t = 0

    def __enter__(self) -> "Fake48kStream":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self, n: int):
        t = np.arange(self._t, self._t + n) / 48000
        self._t += n
        time.sleep(0.005)  # be a polite stream
        return (np.sin(2 * np.pi * 400 * t) * 0.05).astype(np.float32), False


def test_recorder_resamples_48k_stream_to_16k() -> None:
    """THE regression: a device that only accepts 48 kHz (AudioRelay) must
    produce 16 kHz results for the unchanged STT pipeline. A 400 Hz tone
    stays 400 Hz — if resampling were missing, the 48k samples re-read as
    16k would show up at 1200 Hz."""
    rec = VoiceRecorder(stream_factory=Fake48kStream)
    rec.start()
    time.sleep(0.12)
    rec.request_stop()
    res = rec.wait_result(timeout=10)
    assert res.ok and res.audio is not None, f"48k stream must record ok: {res.error}"
    assert res.audio.size >= 4000
    assert abs(_dominant_freq(res.audio) - 400) < 10, (
        f"expected a 400 Hz tone at 16 kHz, got {_dominant_freq(res.audio):.0f} Hz "
        "(missing 48k->16k resample?)"
    )
    print("PASS test_recorder_resamples_48k_stream_to_16k")


class CountingEngine:
    name = "counting"

    def __init__(self) -> None:
        self.total = 0
        self.max_frame = 0

    def add_frame(self, frame) -> str | None:
        self.total += frame.size
        self.max_frame = max(self.max_frame, frame.size)
        return None

    def reset(self) -> None:
        pass


def test_wake_listener_resamples_48k_stream() -> None:
    """The wake engine must only ever see 16-kHz-scaled frames, even when
    the stream runs at 48 kHz (each 4000-sample 48k read -> ~1333 @16k)."""
    eng = CountingEngine()
    listener = WakeListener(eng, on_wake=lambda p: None,
                            stream_factory=lambda: Fake48kStream())
    listener.start()
    time.sleep(0.12)
    listener.stop()
    assert eng.total > 0, "engine must receive frames"
    assert eng.max_frame < 4000, (
        f"engine saw a {eng.max_frame}-sample frame — 48k audio was not "
        "converted to 16k before the engine"
    )
    print("PASS test_wake_listener_resamples_48k_stream")


# ---------------------------------------------------------------------
# 2d. level stats + first-load announcements (the "silent hang" fix)
# ---------------------------------------------------------------------


def test_level_stats_audible_and_silent() -> None:
    from gojo.voice.level import level_stats

    tone = (np.sin(2 * np.pi * 400 * np.arange(16000) / 16000) * 0.1).astype(np.float32)
    st = level_stats(tone)
    assert st["frames"] == 16000 and st["silent"] is False
    assert st["peak"] > 0.05 and st["rms"] > 0.03

    silence = np.zeros(16000, dtype=np.float32)
    st2 = level_stats(silence)
    assert st2["silent"] is True and st2["peak"] == 0.0 and st2["rms"] == 0.0
    assert level_stats(np.zeros(0, dtype=np.float32))["frames"] == 0
    print("PASS test_level_stats_audible_and_silent")


def test_stt_announces_first_model_load_once() -> None:
    """First STT use may download the model — the UI must be told exactly
    once (a first-run download must never look like a frozen 'listening')."""
    from gojo.voice.stt import STTError, WhisperSTT

    notes: list[str] = []
    stt = WhisperSTT(model_size="tiny", on_loading=notes.append)
    for _ in range(2):
        try:
            stt.transcribe(np.zeros(1600, dtype=np.float32))
        except STTError:
            pass  # sandbox: model download unavailable — expected
    assert len(notes) == 1, "first load must be announced exactly once"
    print("PASS test_stt_announces_first_model_load_once")


def test_wake_engine_announces_first_model_load_once() -> None:
    notes: list[str] = []
    eng = WhisperWakeWord(on_loading=notes.append)
    frame = _speaky(4000)
    for _ in range(8):  # one full 2 s window -> first real transcribe attempt
        eng.add_frame(frame)
    assert len(notes) == 1, "first load must be announced exactly once"
    for _ in range(8):
        eng.add_frame(frame)
    assert len(notes) == 1
    print("PASS test_wake_engine_announces_first_model_load_once")


# ---------------------------------------------------------------------
# 2d2. STT config: format identity + language pin (Phase 4 real-PC fix)
# ---------------------------------------------------------------------


def test_recorder_chunk_resample_equals_whole_buffer_resample() -> None:
    """Requirement: the audio the app feeds Whisper (per-250ms-chunk
    resample + concat) must be IDENTICAL in format/range/rate to what
    the doctor feeds Whisper (whole-buffer resample). For 48k->16k (3:1)
    on chunk-aligned buffers the math is exact."""
    sr = 48000
    frame = sr // 4  # 12000 — exactly divisible by 3
    full = (np.sin(2 * np.pi * 400 * np.arange(8 * sr) / sr) * 0.05).astype(np.float32)
    per_chunk = np.concatenate(
        [resample_to_target(full[i:i + frame], sr, 16000)
         for i in range(0, len(full), frame)]
    )
    whole = resample_to_target(full, sr, 16000)
    assert per_chunk.dtype == whole.dtype == np.float32
    assert per_chunk.size == whole.size == 8 * 16000
    assert np.array_equal(per_chunk, whole), "app and doctor audio must be identical"
    # 16k input must pass through untouched (no double conversion)
    buf16 = full[::3]
    assert np.array_equal(resample_to_target(buf16, 16000, 16000), buf16)
    print("PASS test_recorder_chunk_resample_equals_whole_buffer_resample")


def test_stt_language_pin_and_vad_off() -> None:
    """The real-PC fix: STT must (a) accept a language pin ('hi'/'en'/...)
    with auto-detect as default, and (b) run with VAD off (the recorder
    already endpoints; Silero VAD was the prime suspect for empty
    Hindi/Hinglish transcripts)."""
    import inspect
    from gojo.voice.stt import WhisperSTT

    assert WhisperSTT().language == ""  # auto by default
    assert WhisperSTT(language="HI").language == "hi"  # normalized
    src = inspect.getsource(WhisperSTT._real_transcribe)
    assert "vad_filter=False" in src, "app STT must not use Silero VAD"
    assert "self._language or None" in src, "language pin must reach transcribe"
    print("PASS test_stt_language_pin_and_vad_off")


def test_settings_stt_language_env() -> None:
    import os
    from gojo.config import load_settings

    os.environ["GOJO_STT_LANGUAGE"] = "hi"
    try:
        assert load_settings().stt_language == "hi"
    finally:
        del os.environ["GOJO_STT_LANGUAGE"]
    assert load_settings().stt_language == ""
    print("PASS test_settings_stt_language_env")


# ---------------------------------------------------------------------
# 2e. the voice doctor (numbered real-PC report) + deep-dive
# ---------------------------------------------------------------------


class FakeSeg:
    def __init__(self, start, end, text, nsp=0.1, comp=1.1, lp=-0.3):
        self.start, self.end, self.text = start, end, text
        self.no_speech_prob, self.compression_ratio, self.avg_logprob = nsp, comp, lp


class FakeInfo:
    language = "hi"
    language_probability = 0.82


def test_doctor_deep_dive_matrix() -> None:
    """The A/B deep-dive: 4 configs on one sample; VAD-on simulates the
    old empty result, others produce text; per-segment detail + the
    matcher run on the ACTUAL recognized text must all be reported."""
    from gojo.voice.doctor import whisper_deep_dive

    def fake_transcribe(audio, beam_size=1, **kw):
        # mimic the real failure: VAD on -> nothing; else language-dependent text
        if kw.get("vad_filter"):
            return [], FakeInfo()
        lang = kw.get("language") or "auto"
        text = {"hi": "hello gojo", "en": "hey gojo", "auto": "helo gojo"}[lang]
        return [FakeSeg(0.1, 1.2, text)], FakeInfo()

    lines: list[str] = []
    buf16 = np.zeros(16000, dtype=np.float32)
    whisper_deep_dive(fake_transcribe, buf16, "", on_line=lines.append)
    text = "\n".join(lines)
    for needle in (
        "[app config]", "[old app config (VAD on)]", "[language=hi]", "[language=en]",
        "lang=hi p=0.82", "segments=1", "segments=0",
        "text: <empty>", "text: 'hello gojo'",
        "seg0: 0.1-1.2s no_speech=0.10",
        "wake matcher on full text: 'hello gojo'",  # 'helo gojo' -> hello gojo
    ):
        assert needle in text, f"deep-dive report missing {needle!r}:\n{text}"
    print("PASS test_doctor_deep_dive_matrix")


def test_doctor_all_five_phrases_matcher() -> None:
    """Requirement: wake phrase matching must be checked against the
    ACTUAL five phrases (unit-level, deterministic)."""
    for phrase in ("Hey Gojo", "Hi Gojo", "Hello Gojo", "Wake up Gojo", "Yo Gojo"):
        m = match_wake_phrase(phrase)
        assert m is not None, f"{phrase!r} must match"
    print("PASS test_doctor_all_five_phrases_matcher")


def test_doctor_report_with_fakes() -> None:
    from gojo.voice.doctor import analyze_capture, run_doctor

    # pure capture analysis (the [3]-[6] core)
    raw48 = (np.sin(2 * np.pi * 400 * np.arange(48000) / 48000) * 0.05).astype(np.float32)
    a = analyze_capture(raw48, 48000)
    assert a["samples"] == 48000 and a["resampled"] is True
    assert a["samples_16k"] == 16000 and a["silent"] is False

    dev = {"index": 15, "name": "Virtual Mic (AudioRelay)", "channels": 2,
           "default_samplerate": 48000.0}

    class S:
        samplerate = 48000

        def __init__(self) -> None:
            self._t = 0

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def read(self, n):
            t = np.arange(self._t, self._t + n) / 48000
            self._t += n
            time.sleep(0.002)
            return (np.sin(2 * np.pi * 400 * t) * 0.05).astype(np.float32), False

    lines: list[str] = []
    rc = run_doctor(
        seconds=0.25,
        device_spec="15",
        deps={
            "open_stream": lambda spec, sr: (S(), dev, 48000),
            "stt": lambda a16: "hey gojo",
            "wake_engine": lambda: FakeWakeEngine(),
        },
        on_line=lines.append,
    )
    text = "\n".join(lines)
    assert rc == 0, text
    for needle in (
        "Virtual Mic (AudioRelay) (index 15)",   # [1] device
        "48000 Hz",                                # [2] rate
        "48000 Hz -> 16000 Hz",                    # [5] resample yes
        "audible",                                 # [4] verdict
        "transcript: 'hey gojo'",                  # [8] whisper output
        "direct_command=None",                     # [9] brain routing
        "RESULT: all steps OK",
    ):
        assert needle in text, f"report missing {needle!r}:\n{text}"
    # wake probe lines present with frame counts
    assert "[W2]" in text and "frames received" in text
    print("PASS test_doctor_report_with_fakes")


def test_doctor_graceful_without_audio_stack() -> None:
    """Sandbox truth: no PortAudio -> every hardware step FAILs cleanly,
    the report still completes, no traceback, exit code 1."""
    from gojo.voice.doctor import run_doctor

    lines: list[str] = []
    rc = run_doctor(seconds=0.1, device_spec="", on_line=lines.append)
    text = "\n".join(lines)
    assert rc == 1
    assert "FAIL" in text
    assert "Traceback" not in text
    assert "RESULT:" in text
    print("PASS test_doctor_graceful_without_audio_stack")


def test_doctor_inspect_whisper_environment() -> None:
    from gojo.voice.doctor import inspect_whisper_environment

    env = inspect_whisper_environment()
    assert isinstance(env, dict)
    assert "faster_whisper_installed" in env
    assert "ctranslate2_installed" in env
    assert "transcribe_defaults" in env
    assert "temperature_default" in env
    print("PASS test_doctor_inspect_whisper_environment")


def test_doctor_analyze_buffer_forensics() -> None:
    from gojo.voice.doctor import analyze_buffer_forensics

    # Empty
    empty = analyze_buffer_forensics(np.zeros(0, dtype=np.float32))
    assert empty["samples"] == 0 and empty["silent"] is True
    assert empty["verdict"] == "EMPTY BUFFER"

    # Silent
    silent = analyze_buffer_forensics(np.zeros(16000, dtype=np.float32))
    assert silent["samples"] == 16000 and silent["silent"] is True
    assert "SILENT" in silent["verdict"]

    # Normal tone (0.5 amplitude)
    tone = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.5).astype(np.float32)
    normal = analyze_buffer_forensics(tone)
    assert normal["samples"] == 16000 and normal["silent"] is False
    assert normal["clipping_pct"] == 0.0
    assert 0.49 <= normal["peak"] <= 0.51
    assert "NORMAL LEVEL" in normal["verdict"]

    # Clipped (values >= 0.999)
    clipped_arr = np.array([1.0, -1.0, 0.999, -0.999] * 100 + [0.1] * 100, dtype=np.float32)
    clipped = analyze_buffer_forensics(clipped_arr)
    assert clipped["clipping_pct"] > 50.0
    assert "CLIPPING" in clipped["verdict"]
    print("PASS test_doctor_analyze_buffer_forensics")


def test_doctor_save_buffer_to_wav() -> None:
    import wave
    from gojo.voice.doctor import save_buffer_to_wav

    tone = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.5).astype(np.float32)
    wav_path = save_buffer_to_wav(tone)
    try:
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000
    finally:
        if wav_path.exists():
            wav_path.unlink()
    print("PASS test_doctor_save_buffer_to_wav")


def test_doctor_test_mono_stereo_handling() -> None:
    from gojo.voice.doctor import test_mono_stereo_handling

    def mock_transcribe(audio, beam_size=1, **kw):
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            raise ValueError(f"expected 1D array, got ndim={audio.ndim}")
        return [FakeSeg(0.0, 1.0, "test")], FakeInfo()

    lines: list[str] = []
    buf16 = np.zeros(16000, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
    try:
        import wave

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

        res = test_mono_stereo_handling(mock_transcribe, buf16, wav_path=wav_path, on_line=lines.append)
        assert res["1d_mono"]["status"] == "OK"
        assert "REJECTED" in res["2d_mono"]["status"]
        assert "REJECTED" in res["2d_stereo"]["status"]
        assert res["wav_path"]["status"] == "OK"
    finally:
        if wav_path.exists():
            wav_path.unlink()
    print("PASS test_doctor_test_mono_stereo_handling")


def test_doctor_run_stt_deep_matrix() -> None:
    from gojo.voice.doctor import run_stt_deep

    def mock_small_transcribe(audio, **kw):
        lang = kw.get("language") or "en"
        text = "hey gojo" if lang == "en" else "hello gojo"
        if isinstance(kw.get("temperature"), (list, tuple)) and len(kw["temperature"]) > 1:
            time.sleep(0.01)
        return [FakeSeg(0.0, 1.0, text)], FakeInfo()

    def mock_tiny_transcribe(audio, **kw):
        return [FakeSeg(0.0, 0.5, "hey gojo")], FakeInfo()

    lines: list[str] = []
    buf16 = np.zeros(16000, dtype=np.float32)
    report = run_stt_deep(
        small_transcribe=mock_small_transcribe,
        tiny_transcribe=mock_tiny_transcribe,
        buf16=buf16,
        wav_path=None,
        on_line=lines.append,
    )
    text = "\n".join(lines)
    assert len(report["results"]) >= 12
    assert any(r["model"] == "small" for r in report["results"])
    assert any(r["model"] == "tiny" for r in report["results"])
    assert any("temp=0.0" in r["label"] for r in report["results"])
    assert any("temp=(0.0,)" in r["label"] for r in report["results"])
    assert any("temp=fallback" in r["label"] for r in report["results"])
    assert any("no_ts=T" in r["label"] for r in report["results"])
    assert any("beam=5" in r["label"] for r in report["results"])
    assert any("lang=en" in r["label"] for r in report["results"])
    assert any("lang=hi" in r["label"] for r in report["results"])
    assert len(report["findings"]) >= 2
    assert "WAKE MATCH" in text
    print("PASS test_doctor_run_stt_deep_matrix")


def test_doctor_report_with_stt_deep_end_to_end() -> None:
    from gojo.voice.doctor import run_doctor

    dev = {"index": 15, "name": "Virtual Mic (AudioRelay)", "channels": 2,
           "default_samplerate": 48000.0}

    class S:
        samplerate = 48000

        def __init__(self) -> None:
            self._t = 0

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def read(self, n):
            t = np.arange(self._t, self._t + n) / 48000
            self._t += n
            time.sleep(0.002)
            return (np.sin(2 * np.pi * 400 * t) * 0.05).astype(np.float32), False

    def mock_transcribe(audio, **kw):
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            raise ValueError("expected 1D array")
        return [FakeSeg(0.0, 1.0, "hey gojo")], FakeInfo()

    lines: list[str] = []
    rc = run_doctor(
        seconds=0.25,
        device_spec="15",
        stt_deep=True,
        deps={
            "open_stream": lambda spec, sr: (S(), dev, 48000),
            "stt": lambda a16: "hey gojo",
            "wake_engine": lambda: FakeWakeEngine(),
            "model_small": mock_transcribe,
            "model_tiny": mock_transcribe,
        },
        on_line=lines.append,
    )
    text = "\n".join(lines)
    assert rc == 0, text
    for needle in (
        "GOJO STT DEEP-DIVE & AUDIO FORENSICS (--stt-deep)",
        "[A] FASTER-WHISPER ENVIRONMENT INSPECTION:",
        "[B] CAPTURED BUFFER & LOCAL WAV FORENSICS:",
        "[C] INPUT FORMAT & MONO/STEREO HANDLING TEST:",
        "[D] DECODING CONFIGURATIONS MATRIX",
        "[E] DETAILED TRANSCRIPTS",
        "[F] FORENSIC SUMMARY",
        "WAV exported locally to",
        "Digital clipping",
        "RESULT: all steps OK",
    ):
        assert needle in text, f"stt-deep report missing {needle!r}:\n{text}"
    print("PASS test_doctor_report_with_stt_deep_end_to_end")


def test_doctor_main_cli_flags() -> None:
    from gojo.voice.doctor import main

    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    print("PASS test_doctor_main_cli_flags")


# ---------------------------------------------------------------------
# 3. listener: mic handling, callback, clean shutdown
# ---------------------------------------------------------------------


def test_listener_reports_frames_arriving() -> None:
    """The frames_seen counter is the 'are samples actually arriving?'
    signal (no audio is stored — just a count)."""
    listener = WakeListener(FakeWakeEngine(), on_wake=lambda p: None,
                            stream_factory=FakeStreamFactory())
    listener.start()
    time.sleep(0.15)
    assert listener.frames_seen > 0, "frames must arrive while running"
    listener.stop()
    frozen = listener.frames_seen
    time.sleep(0.05)
    assert listener.frames_seen == frozen, "a stopped listener must not receive frames"
    print("PASS test_listener_reports_frames_arriving")


def test_listener_no_mic_is_a_clean_error() -> None:
    factory = FakeStreamFactory()
    factory.fail = True
    got: list[str] = []
    listener = WakeListener(FakeWakeEngine(), on_wake=got.append, stream_factory=factory)
    r = listener.start()
    assert r["ok"] is False and "No microphone" in r["error"]
    assert not listener.running and factory.created == 0
    print("PASS test_listener_no_mic_is_a_clean_error")


def test_listener_fires_callback_and_stops() -> None:
    got: list[str] = []
    listener = WakeListener(
        FakeWakeEngine(trigger_after=3, phrase="hi gojo"),
        on_wake=got.append,
        stream_factory=FakeStreamFactory(),
    )
    assert listener.start()["ok"] is True
    deadline = time.time() + 10
    while not got and time.time() < deadline:
        time.sleep(0.02)
    assert got == ["hi gojo"]
    time.sleep(0.05)  # let the loop exit
    assert not listener.running, "listener must stop itself after a detection"
    print("PASS test_listener_fires_callback_and_stops")


def test_listener_clean_shutdown() -> None:
    listener = WakeListener(FakeWakeEngine(), on_wake=lambda p: None,
                            stream_factory=FakeStreamFactory())
    assert listener.start()["ok"] is True
    assert listener.running
    listener.stop()
    assert not listener.running
    listener.stop()  # idempotent
    listener2 = WakeListener(FakeWakeEngine(), on_wake=lambda p: None,
                             stream_factory=FakeStreamFactory())
    listener2.stop()  # stop without start: no crash
    assert not listener2.running
    print("PASS test_listener_clean_shutdown")


# ---------------------------------------------------------------------
# 4. always-listening: config, enabled/disabled, privacy default
# ---------------------------------------------------------------------


def test_always_listening_default_off_and_privacy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=3)
        assert b["prefs"].get("always_listening") is False
        assert b["pipeline"].listener_running is False
        assert b["factory"].created == 0, "mic must NEVER open by default"
        time.sleep(0.1)
        assert b["factory"].created == 0, "no background listening when off"
    print("PASS test_always_listening_default_off_and_privacy")


def test_always_listening_on_off_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=999)
        p = b["pipeline"]
        p.set_always_listening(True)
        assert b["prefs"].get("always_listening") is True
        assert p.listener_running and b["factory"].created == 1

        # an awake GOJO stops the wake mic (waking is the point)
        b["api"].wake()
        assert p.listener_running is False

        # sleeping resumes it
        b["api"].sleep()
        assert p.listener_running is True

        # toggling off stops it
        b["api"].set_voice_pref("always_listening", False)
        assert b["prefs"].get("always_listening") is False
        assert p.listener_running is False
        # prefs survived in the file (restart-safe)
        reloaded = Prefs(Path(tmp) / "gojo_prefs.json")
        assert reloaded.get("always_listening") is False
    print("PASS test_always_listening_on_off_lifecycle")


def test_always_listening_bad_value_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp))
        r = b["api"].set_voice_pref("always_listening", "yes-please")
        assert r["ok"] is False
        assert b["prefs"].get("always_listening") is False
    print("PASS test_always_listening_bad_value_rejected")


def test_command_toggles_always_listening() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=999)
        b["api"].wake()  # say it while awake (like a normal text/voice turn)

        r = b["api"]._run_turn("gojo, always listening on", via="text")
        assert r["kind"] == "command" and "hamesha sunta" in r["reply"]
        assert b["prefs"].get("always_listening") is True
        # GOJO is awake -> listener waits for sleep (correct), sleep now
        b["api"].sleep()
        assert b["pipeline"].listener_running is True

        r = b["api"]._run_turn("gojo, wake word off", via="text")
        assert r["kind"] == "command"
        assert b["prefs"].get("always_listening") is False
        assert b["pipeline"].listener_running is False
        items = b["transcript"].load()
        assert any(i.get("via") == "system" for i in items), "acks must be in transcript"
    print("PASS test_command_toggles_always_listening")


def test_command_parsing_order() -> None:
    # "wake word on" contains "wake" — must parse as the LISTENING command,
    # not as a plain wake.
    assert detect_direct_command("gojo, wake word on") == "listening_on"
    assert detect_direct_command("wake word off") == "listening_off"
    assert detect_direct_command("always listening on") == "listening_on"
    assert detect_direct_command("always listening on off") is None  # ambiguous
    # existing commands untouched
    assert detect_direct_command("gojo, sleep") == "sleep"
    assert detect_direct_command("wake up") == "wake"
    assert detect_direct_command("always listening") is None  # no on/off
    print("PASS test_command_parsing_order")


# ---------------------------------------------------------------------
# 5. full wake flow: detection -> wake -> ack -> capture -> brain ->
#    TTS -> sleep -> listener re-armed  (state transitions included)
# ---------------------------------------------------------------------


def test_wake_full_flow_state_transitions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=3)
        seen: list[str] = []
        b["state"].add_listener(lambda s: seen.append(s.value))
        p = b["pipeline"]
        p.set_always_listening(True)
        assert p.listener_running

        # wait for the wake to trigger the turn, then for it to finish
        deadline = time.time() + 20
        while not p.busy and time.time() < deadline:
            time.sleep(0.02)
        assert p.busy, "wake turn never started"
        _wait_settled(p)

        # the turn happened: transcript has the voice line + a real reply
        items = b["transcript"].load()
        assert items[0]["role"] == "user" and items[0]["via"] == "voice"
        assert items[0]["text"] == "gojo, hello"
        assert items[1]["role"] == "model"
        # spoken (TTS ran), played, cleaned up
        assert b["tts"].calls == 1
        assert b["played"] and not any(q.exists() for q in b["played"])
        # acknowledgement fired exactly once
        assert b["acks"] == [1]
        assert any(k == "info" and "Gojo?" in m for k, m in b["notes"])
        # ended back at sleep with the LOCAL listener re-armed
        assert b["state"].state is AppState.SLEEPING
        assert p.listener_running is True and b["factory"].created == 2
        # the state machine walked the full wake path
        assert "active" in seen and "listening" in seen and "speaking" in seen
        p.set_always_listening(False)
        assert p.listener_running is False
    print("PASS test_wake_full_flow_state_transitions")


def test_wake_turn_sleep_command_ends_asleep_without_speech() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=3, stt_text="gojo, sleep")
        p = b["pipeline"]
        p.set_always_listening(True)
        deadline = time.time() + 20
        while not p.busy and time.time() < deadline:
            time.sleep(0.02)
        assert p.busy, "wake turn never started"
        _wait_settled(p)
        assert b["state"].state is AppState.SLEEPING
        assert b["tts"].calls == 0, "a sleep ack is not spoken"
        assert p.listener_running is True  # re-armed for the next wake
        p.set_always_listening(False)
    print("PASS test_wake_turn_sleep_command_ends_asleep_without_speech")


def test_wake_ignored_when_busy_or_not_sleeping() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=None)
        p = b["pipeline"]
        # not sleeping: wake callback is a no-op
        b["api"].wake()
        p._on_wake("hey gojo")
        assert b["acks"] == [] and b["state"].state is AppState.ACTIVE
        # busy: wake callback is a no-op
        b["api"].sleep()
        with p._lock:
            p._running = True
        try:
            p._on_wake("hey gojo")
        finally:
            with p._lock:
                p._running = False
        assert b["acks"] == [] and b["state"].state is AppState.SLEEPING
        assert p.listener_running is False
    print("PASS test_wake_ignored_when_busy_or_not_sleeping")


def test_wake_with_no_mic_is_friendly_and_recovers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=None)
        p = b["pipeline"]
        b["factory"].fail = True
        p.set_always_listening(True)
        assert p.listener_running is False
        mic_notes = [m for _k, m in b["notes"] if "mic" in m]
        assert len(mic_notes) == 1, "no-mic must be reported, and only once"
        # simulate mic fix -> next sync starts cleanly
        b["factory"].fail = False
        p.sync_wake()
        assert p.listener_running is True
        p.set_always_listening(False)
    print("PASS test_wake_with_no_mic_is_friendly_and_recovers")


def test_press_to_talk_still_works_alongside_wake() -> None:
    """Phase 3 behavior must be intact when Phase 4 is enabled."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=None)
        p = b["pipeline"]
        p.set_always_listening(True)
        b["api"].wake()  # power button (wake mic auto-stops)
        assert p.listener_running is False
        r = p.start_talk()
        assert r["ok"] is True
        _wait_settled(p)
        items = b["transcript"].load()
        assert items[0]["text"] == "gojo, hello" and items[0]["via"] == "voice"
        # press-to-talk ends ACTIVE (no auto-sleep — that's wake turns only)
        assert b["state"].state is AppState.ACTIVE
        p.set_always_listening(False)
    print("PASS test_press_to_talk_still_works_alongside_wake")


def test_status_exposes_wake_listening() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=999)
        assert b["api"].get_status()["status"]["wake_listening"] is False
        b["pipeline"].set_always_listening(True)
        s = b["api"].get_status()["status"]
        assert s["wake_listening"] is True and s["state"] == "sleeping"
        vs = b["api"].get_voice_settings()
        assert vs["always_listening"] is True and vs["wake_listening"] is True
        b["pipeline"].set_always_listening(False)
    print("PASS test_status_exposes_wake_listening")


# ---------------------------------------------------------------------
# 5b. microphone device surfacing + test (debug aid)
# ---------------------------------------------------------------------


def test_pipeline_test_mic_graceful() -> None:
    """test_mic must ALWAYS return a readable dict — on the user's PC it
    opens the real device for ~1.2 s; here (no PortAudio) it must fail
    gracefully with a reason, never crash."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=None)
        r = b["pipeline"].test_mic()
        assert isinstance(r, dict) and "ok" in r
        if not r["ok"]:
            assert r.get("error"), "failure must carry a readable reason"
        print("PASS test_pipeline_test_mic_graceful")


def test_mic_info_in_voice_settings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=999)
        vs = b["api"].get_voice_settings()
        assert vs["mic"]["configured"] == "system default"
        assert "in_use" in vs["mic"] and "listener_frames" in vs["mic"]
        b["pipeline"].set_always_listening(True)
        time.sleep(0.15)
        vs2 = b["api"].get_voice_settings()
        assert vs2["mic"]["listener_frames"] > 0, "frames counter must move"
        b["pipeline"].set_always_listening(False)
    print("PASS test_mic_info_in_voice_settings")


# ---------------------------------------------------------------------
# 5c. power / sleep button regression (bridge-level path of the UI fix)
# ---------------------------------------------------------------------


def test_power_semantics_sleep_path() -> None:
    """The fixed power-button path, at the bridge level:
    - awake + listener ON  -> sleep()  -> sleeping WITH listener (wake listening)
    - 'power' from wake listening -> set_voice_pref(always_listening, False)
      -> FULLY DOWN (listener stopped, state sleeping)
    - 'power' from fully down -> wake() -> active
    - awake + listener OFF -> sleep() -> fully down
    """
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=999)
        api, state, p = b["api"], b["state"], b["pipeline"]

        api.wake()
        p.set_always_listening(True)
        api.sleep()
        assert state.state is AppState.SLEEPING and p.listener_running is True

        r = api.set_voice_pref("always_listening", False)  # the power-button call
        assert r["ok"] is True
        assert state.state is AppState.SLEEPING
        assert p.listener_running is False, "wake listening must stop on power-off"
        assert b["prefs"].get("always_listening") is False

        api.wake()
        assert state.state is AppState.ACTIVE and p.listener_running is False

        api.sleep()
        assert state.state is AppState.SLEEPING and p.listener_running is False
    print("PASS test_power_semantics_sleep_path")


def test_sleep_command_still_works_with_listener_on() -> None:
    """'gojo, sleep' via the brain path while opted in: ends sleeping with
    the listener re-armed (wake listening), not stuck anywhere."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _build(Path(tmp), trigger_after=None, stt_text="gojo, sleep")
        api, state, p = b["api"], b["state"], b["pipeline"]
        p.set_always_listening(True)
        api.wake()
        r = api._run_turn("gojo, sleep", via="text")
        assert r["kind"] == "command"
        assert state.state is AppState.SLEEPING
        assert p.listener_running is True  # re-armed: still opted in
        p.set_always_listening(False)
    print("PASS test_sleep_command_still_works_with_listener_on")


# ---------------------------------------------------------------------
# 6. real detector (runs where the tiny model can load; honest skip otherwise)
# ---------------------------------------------------------------------


def test_real_engine_silence_and_noise_never_wake() -> None:
    """Loads the REAL tiny model and checks that silence and a steady
    tone never produce a wake. On the user's PC this also verifies the
    model download + int8 CPU inference path. Skipped (not failed) where
    the model cannot be loaded (e.g. offline sandbox)."""
    try:
        eng = WhisperWakeWord()  # real engine, no injection
        eng._get_model()  # force the one-time download + load now (not inside add_frame)
    except Exception as exc:  # noqa: BLE001 — model not loadable here
        print(f"PASS test_real_engine_silence_and_noise_never_wake "
              f"(skipped: real model not loadable here: {type(exc).__name__})")
        return
    frame_n = 4000
    for _ in range(12):  # 3 s silence (skips ASR via the RMS gate)
        assert eng.add_frame(np.zeros(frame_n, dtype=np.float32)) is None
    # a steady 440 Hz tone: crosses the RMS gate -> real inference runs
    tone = (np.sin(np.linspace(0, 220 * np.pi, frame_n)) * 0.015).astype(np.float32)
    for _ in range(12):
        assert eng.add_frame(tone) is None, "a tone must never wake GOJO"
    print("PASS test_real_engine_silence_and_noise_never_wake")


def main() -> int:
    tests = [
        test_matcher_required_phrases,
        test_matcher_asr_variants,
        test_matcher_rejects_invalid_input,
        test_engine_windowing_and_match,
        test_engine_silence_never_transcribes,
        test_engine_tolerates_transcribe_errors,
        test_engine_model_size_is_capped,
        test_resolve_input_device_pure,
        test_resample_48k_to_16k,
        test_resample_44100_to_16k,
        test_choose_stream_rate,
        test_recorder_resamples_48k_stream_to_16k,
        test_wake_listener_resamples_48k_stream,
        test_level_stats_audible_and_silent,
        test_stt_announces_first_model_load_once,
        test_wake_engine_announces_first_model_load_once,
        test_recorder_chunk_resample_equals_whole_buffer_resample,
        test_stt_language_pin_and_vad_off,
        test_settings_stt_language_env,
        test_doctor_deep_dive_matrix,
        test_doctor_all_five_phrases_matcher,
        test_doctor_report_with_fakes,
        test_doctor_graceful_without_audio_stack,
        test_doctor_inspect_whisper_environment,
        test_doctor_analyze_buffer_forensics,
        test_doctor_save_buffer_to_wav,
        test_doctor_test_mono_stereo_handling,
        test_doctor_run_stt_deep_matrix,
        test_doctor_report_with_stt_deep_end_to_end,
        test_doctor_main_cli_flags,
        test_listener_no_mic_is_a_clean_error,
        test_listener_reports_frames_arriving,
        test_listener_fires_callback_and_stops,
        test_listener_clean_shutdown,
        test_always_listening_default_off_and_privacy,
        test_always_listening_on_off_lifecycle,
        test_always_listening_bad_value_rejected,
        test_command_toggles_always_listening,
        test_command_parsing_order,
        test_wake_full_flow_state_transitions,
        test_wake_turn_sleep_command_ends_asleep_without_speech,
        test_wake_ignored_when_busy_or_not_sleeping,
        test_wake_with_no_mic_is_friendly_and_recovers,
        test_press_to_talk_still_works_alongside_wake,
        test_status_exposes_wake_listening,
        test_pipeline_test_mic_graceful,
        test_mic_info_in_voice_settings,
        test_power_semantics_sleep_path,
        test_sleep_command_still_works_with_listener_on,
        test_real_engine_silence_and_noise_never_wake,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Checks for the Phase 3 voice layer: recorder, STT, and the full
press-to-talk pipeline (with fakes — no mic, no network, no model).

Run:  python -m tests.test_voice
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from gojo.ai.mock import MockProvider
from gojo.config import Settings
from gojo.core.state import AppState, StateManager
from gojo.prefs import Prefs
from gojo.transcript import TranscriptStore
from gojo.ui.api import GojoAPI
from gojo.voice.pipeline import VoicePipeline
from gojo.voice.playback import AudioPlayer
from gojo.voice.recorder import VoiceRecorder
from gojo.voice.stt import STTError, WhisperSTT
from gojo.voice.tts.base import TTSError, TTSProvider
from gojo.voice.tts.router import TTSRouter


# ---------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------


class FakeStream:
    """Stands in for sounddevice.InputStream. Plays `speech` frames (each
    250 ms) then silence forever — so the endpoint detector can fire."""

    def __init__(self, sr: int, speech_frames: int = 16) -> None:
        self._sr = sr
        self._frame = sr // 4
        self._left_speech = speech_frames
        self._closed = False

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc) -> None:
        self._closed = True

    def read(self, n: int):
        if self._left_speech > 0:
            self._left_speech -= 1
            data = np.sin(np.linspace(0, 40 * np.pi, n)).astype(np.float32) * 0.1
        else:
            data = np.zeros(n, dtype=np.float32)
            time.sleep(0.01)  # be a polite stream
        return data, False


class ScriptedRecorder:
    """For pipeline tests: no threads, deterministic results."""

    def __init__(self, result) -> None:
        self._result = result
        self.stops = 0

    def start(self) -> None:
        pass

    def request_stop(self) -> None:
        self.stops += 1

    def wait_result(self, timeout: float | None = None):
        time.sleep(0.2)  # keep a stable "busy" window for double-start checks
        return self._result


class FileTTS(TTSProvider):
    """Writes a fake mp3 and records what text it was given."""

    def __init__(self, name: str = "fish", fail: bool = False) -> None:
        self.name = name
        self._fail = fail
        self.got_text: list[str] = []
        self.calls = 0

    def synthesize(self, text: str, out_dir: Path) -> Path:
        self.calls += 1
        self.got_text.append(text)
        if self._fail:
            raise TTSError(f"{self.name} down", provider=self.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"fake-{self.calls}.mp3"
        path.write_bytes(b"ID3-fake")
        return path


def _recording_result(audio: np.ndarray) -> object:
    from gojo.voice.recorder import RecordingResult

    return RecordingResult(ok=True, audio=audio, duration=len(audio) / 16000, stopped_by="silence")


# ---------------------------------------------------------------------
# recorder (real thread + fake stream)
# ---------------------------------------------------------------------


def test_recorder_auto_stops_on_silence() -> None:
    rec = VoiceRecorder(
        stream_factory=lambda: FakeStream(16000, speech_frames=16),
        silence_seconds=0.5,
        min_seconds=0.4,
        max_seconds=20,
    )
    rec.start()
    result = rec.wait_result(timeout=30)
    assert result.ok, f"recording failed: {result.error}"
    assert result.stopped_by == "silence"
    assert result.audio is not None and len(result.audio) > 16000
    print("PASS test_recorder_auto_stops_on_silence")


def test_recorder_manual_stop() -> None:
    rec = VoiceRecorder(
        stream_factory=lambda: FakeStream(16000, speech_frames=500),  # lots of "speech"
        silence_seconds=99,  # endpoint must NOT fire
        max_seconds=99,
    )
    rec.start()
    time.sleep(0.6)
    rec.request_stop()
    result = rec.wait_result(timeout=30)
    assert result.ok and result.stopped_by == "user"
    print("PASS test_recorder_manual_stop")


def test_recorder_no_mic_is_a_clean_error() -> None:
    def no_mic():
        raise RuntimeError("default input device not found")

    rec = VoiceRecorder(stream_factory=no_mic)
    rec.start()
    result = rec.wait_result(timeout=30)
    assert result.ok is False
    assert result.error and "No microphone" in result.error
    print("PASS test_recorder_no_mic_is_a_clean_error")


# ---------------------------------------------------------------------
# stt
# ---------------------------------------------------------------------


def test_stt_injects_transcript() -> None:
    stt = WhisperSTT(model_size="small", transcribe_fn=lambda audio: "  gojo, hello  ")
    assert stt.transcribe(np.zeros(16000, dtype=np.float32)) == "gojo, hello"
    print("PASS test_stt_injects_transcript")


def test_stt_wraperrors() -> None:
    def broken(audio):
        raise RuntimeError("model exploded")

    stt = WhisperSTT(transcribe_fn=broken)
    try:
        stt.transcribe(np.zeros(16000, dtype=np.float32))
    except STTError as exc:
        assert "model exploded" in str(exc)
        return
    raise AssertionError("raw error escaped as non-STTError")
    print("PASS test_stt_wraperrors")


def test_stt_validates_size() -> None:
    assert WhisperSTT(model_size="huge-model").model_size == "small"
    assert WhisperSTT(model_size="base").model_size == "base"
    print("PASS test_stt_validates_size")


# ---------------------------------------------------------------------
# full press-to-talk pipeline (fakes end to end)
# ---------------------------------------------------------------------


def _build_pipeline(tmp: Path, *, transcript_text: str, tts: TTSProvider,
                    autoplay: bool = True, awake: bool = True,
                    stt_error: bool = False):
    settings = Settings(
        gemini_api_key=None, gemini_model="mock-model", provider="mock",
        data_dir=tmp, log_level="INFO",
    )
    state = StateManager(initial=AppState.SLEEPING)
    if awake:
        state.wake()
    transcript = TranscriptStore(tmp)
    prefs = Prefs(tmp / "gojo_prefs.json",
                  defaults={"tts_autoplay": autoplay})
    api = GojoAPI(settings=settings, brain=MockProvider(), state=state,
                  transcript=transcript, version="t", prefs=prefs)
    notes: list[tuple[str, str]] = []
    rec = np.zeros(16000, dtype=np.float32)

    def _boom(audio):
        raise RuntimeError("boom")

    stt = WhisperSTT(
        transcribe_fn=_boom if stt_error else (lambda audio: transcript_text)
    )
    played: list[Path] = []
    pipeline = VoicePipeline(
        state=state,
        recorder=ScriptedRecorder(_recording_result(rec)),
        stt=stt,
        tts=TTSRouter([tts], order="fish"),
        player=AudioPlayer(play_fn=lambda p: played.append(p)),
        on_text=api._run_turn,
        prefs=prefs,
        on_note=lambda kind, msg: notes.append((kind, msg)),
        audio_dir=tmp / "audio",
    )
    api.attach_voice(pipeline)
    return api, state, transcript, notes, played, pipeline


def _run_turn(pipeline) -> None:
    assert pipeline.start_talk()["ok"] is True
    deadline = time.time() + 20
    while pipeline.busy and time.time() < deadline:
        time.sleep(0.05)
    assert not pipeline.busy, "voice turn did not finish"


def test_pipeline_full_happy_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tts = FileTTS()
        api, state, transcript, notes, played, pipeline = _build_pipeline(
            Path(tmp), transcript_text="gojo, hello", tts=tts
        )
        _run_turn(pipeline)

        # transcript: user line marked voice + real mock reply
        items = transcript.load()
        assert items[0]["role"] == "user" and items[0]["via"] == "voice"
        assert items[0]["text"] == "gojo, hello"
        assert items[1]["role"] == "model" and "gojo, hello" in items[1]["text"]

        # TTS got the reply, playback happened, file cleaned up
        assert tts.calls == 1
        assert played, "player was never called"
        assert not any(p.exists() for p in played), "temp audio must be deleted"

        # back to idle (active), nothing left busy
        assert state.state is AppState.ACTIVE
        assert api.get_status()["status"]["busy"] is False
    print("PASS test_pipeline_full_happy_path")


def test_pipeline_speaks_command_acks_too() -> None:
    """'gojo, sleep' BY VOICE: command detected, GOJO sleeps (no TTS —
    a sleeping GOJO doesn't talk)."""
    with tempfile.TemporaryDirectory() as tmp:
        tts = FileTTS()
        api, state, transcript, notes, played, pipeline = _build_pipeline(
            Path(tmp), transcript_text="gojo, sleep", tts=tts
        )
        _run_turn(pipeline)
        assert state.state is AppState.SLEEPING
        assert tts.calls == 0 and played == []
    print("PASS test_pipeline_speaks_command_acks_too")


def test_pipeline_autoplay_off_stays_silent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tts = FileTTS()
        api, state, transcript, notes, played, pipeline = _build_pipeline(
            Path(tmp), transcript_text="gojo, hello", tts=tts, autoplay=False
        )
        _run_turn(pipeline)
        assert tts.calls == 0 and played == []
        assert state.state is AppState.ACTIVE
        assert transcript.load()[1]["role"] == "model"  # reply still in transcript
    print("PASS test_pipeline_autoplay_off_stays_silent")


def test_pipeline_empty_transcript_is_friendly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tts = FileTTS()
        api, state, transcript, notes, played, pipeline = _build_pipeline(
            Path(tmp), transcript_text="   ", tts=tts
        )
        _run_turn(pipeline)
        assert state.state is AppState.ACTIVE
        assert any("didn't catch that" in m for _k, m in notes)
        assert transcript.load() == []  # nothing pollutes the transcript
    print("PASS test_pipeline_empty_transcript_is_friendly")


def test_pipeline_stt_failure_is_friendly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tts = FileTTS()
        api, state, transcript, notes, played, pipeline = _build_pipeline(
            Path(tmp), transcript_text="x", tts=tts, stt_error=True
        )
        _run_turn(pipeline)
        assert state.state is AppState.ACTIVE
        assert any("didn't catch that" in m for _k, m in notes)
    print("PASS test_pipeline_stt_failure_is_friendly")


def test_pipeline_tts_fallback_note() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fish = FileTTS(name="fish", fail=True)
        windows = FileTTS(name="windows")
        settings = Settings(None, "mock-model", "mock", Path(tmp), "INFO")
        state = StateManager(initial=AppState.ACTIVE)
        transcript = TranscriptStore(Path(tmp))
        prefs = Prefs(Path(tmp) / "gojo_prefs.json")
        api = GojoAPI(settings=settings, brain=MockProvider(), state=state,
                      transcript=transcript, version="t", prefs=prefs)
        notes: list = []
        pipeline = VoicePipeline(
            state=state,
            recorder=ScriptedRecorder(_recording_result(np.zeros(16000, dtype=np.float32))),
            stt=WhisperSTT(transcribe_fn=lambda a: "gojo, hello"),
            tts=TTSRouter([fish, windows], order="auto"),
            player=AudioPlayer(play_fn=lambda p: None),
            on_text=api._run_turn,
            prefs=prefs,
            on_note=lambda kind, msg: notes.append((kind, msg)),
            audio_dir=Path(tmp) / "audio",
        )
        api.attach_voice(pipeline)
        _run_turn(pipeline)
        assert fish.calls == 1 and windows.calls == 1, "fallback must be used"
        assert any("backup voice" in m for _k, m in notes), "fallback note missing"
    print("PASS test_pipeline_tts_fallback_note")


def test_pipeline_start_blocked_when_asleep() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tts = FileTTS()
        api, state, transcript, notes, played, pipeline = _build_pipeline(
            Path(tmp), transcript_text="hi", tts=tts, awake=False
        )
        r = pipeline.start_talk()
        assert r["ok"] is False and "not awake" in r["error"]
        assert state.state is AppState.SLEEPING
    print("PASS test_pipeline_start_blocked_when_asleep")


def test_pipeline_double_start_blocked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tts = FileTTS()
        api, state, transcript, notes, played, pipeline = _build_pipeline(
            Path(tmp), transcript_text="hi", tts=tts
        )
        r1 = pipeline.start_talk()
        r2 = pipeline.start_talk()
        assert r1["ok"] is True, r1
        assert r2["ok"] is False and "already" in r2["error"], r2
        deadline = time.time() + 20
        while pipeline.busy and time.time() < deadline:
            time.sleep(0.05)
        assert not pipeline.busy
        # after the first turn finishes, a new turn must be possible again
        assert pipeline.start_talk()["ok"] is True
        deadline = time.time() + 20
        while pipeline.busy and time.time() < deadline:
            time.sleep(0.05)
    print("PASS test_pipeline_double_start_blocked")


def main() -> int:
    tests = [
        test_recorder_auto_stops_on_silence,
        test_recorder_manual_stop,
        test_recorder_no_mic_is_a_clean_error,
        test_stt_injects_transcript,
        test_stt_wraperrors,
        test_stt_validates_size,
        test_pipeline_full_happy_path,
        test_pipeline_speaks_command_acks_too,
        test_pipeline_autoplay_off_stays_silent,
        test_pipeline_empty_transcript_is_friendly,
        test_pipeline_stt_failure_is_friendly,
        test_pipeline_tts_fallback_note,
        test_pipeline_start_blocked_when_asleep,
        test_pipeline_double_start_blocked,
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

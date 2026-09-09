"""Wake-word detection (Phase 4) — LOCAL ONLY, user-opt-in, privacy-safe.

Phrases: "Hey Gojo", "Hi Gojo", "Hello Gojo" (matched fuzzy, ASR-tolerant).

How it works
------------
A small local speech recognizer (faster-whisper "tiny", int8 on CPU —
already part of GOJO's STT stack, no new heavy dependency) transcribes
short sliding 2 s windows of microphone audio once per second. Each
transcript is matched against the wake phrases by pure string logic.
Silent windows are skipped before transcription (RMS gate), so the CPU
cost is near zero when nobody is talking.

Why not openWakeWord? (due diligence, checked 2026-09)
- Its pre-trained models are alexa / hey mycroft / hey jarvis /
  hey rhasspy / weather / timers — NO "Gojo" phrase exists.
- It is English-only (training data is English synthetic speech); the
  user speaks Hinglish.
- A custom "hey gojo" model needs its full training pipeline
  (espeak-ng on Windows + hours of training) and its quality could not
  be verified without real microphone testing — we don't ship
  unverified detectors.
- faster-whisper's tiny model already runs on this exact stack
  (verified cp314/Windows wheels in Phase 3) and understands Hindi and
  English, so custom phrases cost zero training.

Privacy
-------
- The listener is OFF by default and only runs when the user explicitly
  enables "always listening" (settings panel or "Gojo, always listening on").
- Everything here is local: no audio, transcript, or frame ever leaves
  the PC. Wake detection never touches Gemini or Fish Audio — the shared
  brain/TTS path only runs AFTER a wake, on the captured command.

Injectability (tests)
---------------------
- `transcribe_fn` — stand-in for Whisper inference
- `WakeListener(stream_factory=...)` — stand-in for the microphone
  (same pattern as VoiceRecorder)
"""
from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from typing import Callable, Optional

import numpy as np

from .resample import resample_to_target

logger = logging.getLogger("gojo.voice.wakeword")

WAKE_PHRASES = (
    "hey gojo",
    "hi gojo",
    "hello gojo",
    "wake up gojo",
    "yo gojo",
)

# Greeting variants Whisper tends to produce for the wake phrases.
_WAKE_GREETINGS = frozenset(
    {"hey", "hi", "hello", "hii", "hiiiii", "hay", "haio", "he", "helo", "hais", "yo"}
)
_GOJO_SPLIT_A = frozenset({"go", "goh", "goy"})  # "go jo" — split across tokens
_GOJO_SPLIT_B = frozenset({"jo"})


# ---------------------------------------------------------------------
# phrase matching — pure string logic, fully unit-testable
# ---------------------------------------------------------------------


def _levenshtein(a: str, b: str, cap: int = 1) -> int:
    """Edit distance, bailing out early once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            row_min = min(row_min, v)
        if row_min > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _normalize(transcript: str) -> str:
    t = unicodedata.normalize("NFKC", transcript or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)  # punctuation -> spaces, unicode words kept
    return " ".join(t.split())


def _canonical_phrase(tokens: list[str], idx: int) -> str:
    """Map a found "gojo" to the nearest canonical wake phrase, using a
    greeting within the two preceding tokens when one exists."""
    window = tokens[max(0, idx - 2) : idx]
    if "wake" in window:
        return "wake up gojo"
    for g in reversed(window):
        if g in _WAKE_GREETINGS:
            if g.startswith("hello") or g == "helo":
                return "hello gojo"
            if g.startswith("hi"):
                return "hi gojo"
            if g == "yo":
                return "yo gojo"
            return "hey gojo"
    return "hey gojo"  # greeting optional — ASR drops it often


def match_wake_phrase(transcript: str) -> Optional[str]:
    """Return the canonical wake phrase ("hey gojo" | "hi gojo" |
    "hello gojo") if an ASR transcript contains one, else None.

    Tolerates the common Whisper variants for a foreign name:
    punctuation, case, "goyo"/"goho"/"gojo" (edit distance 1),
    "go jo" split across tokens, and fused forms like "heygojo".
    A bare "gojo" (no greeting) counts — the name is a strong signal.
    """
    t = _normalize(transcript)
    if not t:
        return None
    tokens = t.split()

    for i, tok in enumerate(tokens):
        if tok == "gojo":
            return _canonical_phrase(tokens, i)
        if "gojo" in tok and len(tok) <= 8:  # fused: "heygojo"
            return _canonical_phrase(tokens, i)
        if len(tok) >= 3 and _levenshtein(tok, "gojo") <= 1:  # goyo, goho, gojh...
            return _canonical_phrase(tokens, i)

    for i in range(len(tokens) - 1):  # "go jo"
        if tokens[i] in _GOJO_SPLIT_A and tokens[i + 1] in _GOJO_SPLIT_B:
            return _canonical_phrase(tokens, i)
    return None


# ---------------------------------------------------------------------
# engine: sliding-window local ASR + matcher
# ---------------------------------------------------------------------


class WakeWordEngine:
    """Interface: feed 16 kHz mono float32 frames, get a matched phrase.

    Implementations must be thread-confined (one consumer thread) and
    must never raise out of add_frame — detection is best-effort.
    """

    name = "base"

    def add_frame(self, frame: np.ndarray) -> Optional[str]:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any buffered audio (e.g. after a detection)."""


class WhisperWakeWord(WakeWordEngine):
    """Real local detector: faster-whisper "tiny" over sliding windows.

    window_seconds of audio is transcribed every step_seconds; a match
    resets the buffer and enters a cooldown so the same utterance cannot
    fire twice. Silent windows (RMS below the gate) skip transcription —
    that is what keeps CPU near zero on an idle room.
    """

    name = "whisper"

    def __init__(
        self,
        model_size: str = "tiny",
        transcribe_fn: Optional[Callable[[np.ndarray], str]] = None,
        matcher: Callable[[str], Optional[str]] = match_wake_phrase,
        sr: int = 16000,
        window_seconds: float = 2.0,
        step_seconds: float = 1.0,
        silence_rms: float = 0.006,
        cooldown_seconds: float = 3.0,
        hotwords: str = "gojo",
    ) -> None:
        if model_size not in ("tiny", "base"):
            model_size = "tiny"  # CPU budget: never load a big model here
        self._model_size = model_size
        self._inject = transcribe_fn
        self._matcher = matcher
        self._sr = sr
        self._window_n = int(sr * window_seconds)
        self._step_n = max(1, int(sr * step_seconds))
        self._silence_rms = silence_rms
        self._cooldown = cooldown_seconds
        self._hotwords = hotwords
        self._buf = np.zeros(0, dtype=np.float32)
        self._model = None  # lazy — first transcribe only
        self._last_match = 0.0

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy: lazy

            logger.info("loading wake-word model '%s' (one-time download on first use)",
                        self._model_size)
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        return self._model

    def _transcribe(self, chunk: np.ndarray) -> str:
        if self._inject is not None:
            return self._inject(chunk)
        model = self._get_model()
        segments, _info = model.transcribe(
            chunk,
            language=None,  # auto-detect (Hindi/English/Hinglish)
            beam_size=1,    # fastest, plenty for 2 s windows
            vad_filter=False,  # the window is short; our RMS gate already filters silence
            condition_on_previous_text=False,  # windows are independent
            hotwords=self._hotwords or None,  # bias decoding toward "gojo"
        )
        return " ".join(s.text for s in segments).strip()

    def add_frame(self, frame: np.ndarray) -> Optional[str]:
        try:
            frame = np.asarray(frame, dtype=np.float32).flatten()
            self._buf = np.concatenate([self._buf, frame]) if self._buf.size else frame
            while self._buf.size >= self._window_n:
                chunk, self._buf = self._buf[: self._window_n], self._buf[self._step_n :]
                if time.time() - self._last_match < self._cooldown:
                    continue
                rms = float(np.sqrt(np.mean(chunk * chunk)))
                if rms < self._silence_rms:
                    continue  # nobody talking — skip ASR entirely
                try:
                    text = self._transcribe(chunk)
                except Exception:  # noqa: BLE001 — detection is best-effort
                    logger.exception("wake transcription failed")
                    continue
                matched = self._matcher(text)
                if matched:
                    self._last_match = time.time()
                    self._buf = np.zeros(0, dtype=np.float32)
                    logger.info("wake phrase matched from transcript %r", text)
                    return matched
            return None
        except Exception:  # noqa: BLE001 — NEVER let the listener thread die
            logger.exception("wake engine frame failed")
            return None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------
# listener: owns the microphone + one consumer thread
# ---------------------------------------------------------------------


class WakeListener:
    """Runs the engine on a background thread while the user opted in.

    `on_wake(phrase)` is called FROM THIS THREAD when a phrase matches;
    the listener then stops itself (the caller starts the command
    capture). start()/stop() are safe to call from any thread and are
    idempotent.
    """

    def __init__(
        self,
        engine: WakeWordEngine,
        on_wake: Callable[[str], None],
        stream_factory: Optional[Callable[[], object]] = None,
        sr: int = 16000,
        frame_ms: int = 250,
        device_spec: str = "",
    ) -> None:
        self._engine = engine
        self._on_wake = on_wake
        self._stream_factory = stream_factory
        self._sr = sr
        self._frame = max(1, int(sr * frame_ms / 1000))
        self._device_spec = (device_spec or "").strip()  # GOJO_MIC_DEVICE
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_device = ""
        self._frames_seen = 0

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def last_device(self) -> str:
        """Name+index of the device the last real listener used ('' if none
        yet or a fake stream factory was injected)."""
        return self._last_device

    @property
    def frames_seen(self) -> int:
        """Audio frames received by the (last) listener run — proof that
        samples are actually arriving from the device."""
        return self._frames_seen

    def start(self) -> dict:
        """Open the mic and start detecting. Returns {"ok": bool, ...}."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": True}
            self._stop_event.clear()
        self._engine.reset()
        try:
            if self._stream_factory is not None:
                stream = self._stream_factory()
            else:
                from . import devices

                stream, dev, stream_sr = devices.open_input_stream(self._device_spec, self._sr)
                stream_sr = int(getattr(stream, "samplerate", 0) or stream_sr)
                self._last_device = f"{dev['name']} (index {dev['index']}, {stream_sr} Hz)"
        except Exception as exc:  # noqa: BLE001 — no mic / no PortAudio / blocked
            return {"ok": False, "error": f"No microphone available: {exc}"}
        self._frames_seen = 0
        thread = threading.Thread(
            target=self._run, args=(stream,), name="gojo-wakeword", daemon=True
        )
        with self._lock:
            self._thread = thread
        thread.start()
        logger.info(
            "wake listener started (local-only) on: %s",
            self._last_device or "injected stream (test)",
        )
        return {"ok": True}

    def _run(self, stream) -> None:
        try:
            with stream:
                # the stream may run at the device's native rate (e.g. 48 kHz
                # on AudioRelay); the engine always expects 16 kHz frames
                stream_sr = int(getattr(stream, "samplerate", 0) or self._sr)
                while not self._stop_event.is_set():
                    data, _overflow = stream.read(self._frame)
                    self._frames_seen += 1
                    if self._stop_event.is_set():
                        break
                    arr = np.asarray(data).flatten()
                    if stream_sr != self._sr:
                        arr = resample_to_target(arr, stream_sr, self._sr)
                    phrase = self._engine.add_frame(arr)
                    if phrase:
                        try:
                            self._on_wake(phrase)
                        except Exception:  # noqa: BLE001
                            logger.exception("wake callback failed")
                        break
        except Exception:  # noqa: BLE001 — stream error: stop quietly, log detail
            logger.exception("wake listener stream error")
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None

    def stop(self) -> None:
        """Stop detection and release the microphone (idempotent)."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        with self._lock:
            self._thread = None


def build_wakeword_engine() -> WakeWordEngine:
    """The real local engine. Cheap to construct (the model itself loads
    lazily on the first non-silent window)."""
    return WhisperWakeWord()

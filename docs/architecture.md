# GOJO — Architecture Map

Updated: Phase 4 (wake word). Read this before adding anything new.

## The one rule

**Everything talks to the core through small interfaces. No component
reaches around another.** The UI never touches the brain; the brain never
knows a UI exists; the state machine doesn't care who asked.

```
                    ┌────────────────────────────────────────────┐
                    │                GOJO CORE                   │
                    │                                            │
  CLI (Phase 1) ───▶│  ai/router ──▶ ai/gemini (or any provider) │
                    │        │           (prompts.py: personality)
  Desktop UI ──────▶│  core/state   (sleeping / active / thinking
  (pywebview window)│                   / listening / speaking /
   via ui/api.py ──▶│                    error)
                    │  core/commands (sleep / wake detection)
                    │  transcript   (JSONL sessions → SQLite in Ph.6,
                    │                every line tagged with its origin)
                    │  config       (.env — secrets live here only)
                    │  prefs        (data/gojo_prefs.json — UI toggles)
                    │                                            │
  Voice (Phase 3) ─▶│  voice/pipeline:                          │
  (Talk button)     │    recorder → stt → ★ shared brain path ★ │
   (ui/api.py)      │    → tts/router → playback                │
  Wake (Phase 4) ──▶│  voice/wakeword (LOCAL, opt-in):          │
  (opt-in listener) │    tiny-Whisper windows → fuzzy "Hey Gojo"│
                    │    → pipeline._on_wake → SAME brain path  │
                    └────────────────────────────────────────────┘
```

★ = the ONE brain path. Text and voice both enter
`GojoAPI._run_turn(text, via=...)`; there is no second chat implementation.

## Phase 3 decisions (and why)

### One brain path — voice is a transport, not a new system
`VoicePipeline._turn` records, transcribes, then calls the exact
`api._run_turn` that `chat()` uses: command detection → brain → transcript.
So "gojo, sleep" spoken works instantly (command short-circuit), the
personality prompt is unchanged, and a voice conversation is literally the
same transcript as a typed one — tagged `via: "voice"` (no separate voice
history, per spec).

### Mic: press-to-talk only, with endpointing
`voice/recorder.py` opens a sounddevice stream ONLY while a turn is running
(16 kHz mono float32). End-of-speech = RMS below ~0.006 for 1.2 s (with a
0.8 s minimum so a one-word answer doesn't cut itself off, and a 60 s hard
cap). The Talk button can also stop early (`request_stop`). The mic stream
is closed when the turn ends — at no point does GOJO listen continuously,
and no camera access exists or is introduced in this phase.
Wake word is Phase 4: `voice/wakeword.py` defines the `WakeWordEngine`
interface (candidates: openWakeWord / Porcupine) and `build_wakeword_engine()`
returns `None` today — nothing imports a continuous mic path.

### STT: faster-whisper, local, lazy, injectable
`voice/stt.py` wraps faster-whisper: `compute_type="int8"`,
`device="cpu"` (user has no GPU), `vad_filter=True`, `beam_size=1`,
`language=None` → automatic language detection (Hindi/English/Hinglish).
Model size from `.env` (`GOJO_STT_MODEL`, default `small` ≈ 460 MB —
deliberately NOT the big models, per the 16 GB / no-GPU budget). The model
imports and loads on first use, and `transcribe_fn` is injectable so every
test runs with fakes — CI/sandbox never needs ctranslate2 or a model.

### TTS: provider chain with an automatic, honest fallback
`voice/tts/base.py` defines `TTSProvider.synthesize(text, out_dir) -> Path`
plus `TTSError(retryable=...)` / `TTSResult(fell_back=...)`.
`TTSRouter` orders providers from settings (`auto`/`fish` → Fish then local;
`windows` → local only) and falls through on retryable or any failure —
a Fish outage never crashes a voice turn. `FishAudioTTS` implements the
verified v1 contract (POST `https://api.fish.audio/v1/tts`, Bearer key,
`model:` header, `{text, reference_id, format}` body, raw MP3 bytes back);
its `available` is False for missing/placeholder keys, and `post` is
injectable for tests. `WindowsTTS` (pyttsx3/SAPI) is the offline backstop.
Per spec the fallback is a BACKUP, never the default, while Fish is
configured — and the UI shows a one-time note when the fallback was used.

### Speech normalization is a TTS-only layer
`voice/tts/normalize.py` runs before synthesis only: NFKC, markdown
stripped, `C++` → "C plus plus", `₹/Rs/INR` → "rupees", emoji removed,
plus a `DEFAULT_OVERRIDES` table for future term quirks. Deliberate
non-goals: no number→words expansion, no rewriting of technical terms
(VS Code, Python, GitHub, API, Android, SaBuddy stay). The visible
transcript is never touched.

### Playback and audio hygiene
`voice/playback.py` plays MP3 via `winsound.PlaySound` (async mode — the UI
thread never blocks, no console window). The temp file is deleted in a
`finally`, and `sweep_stale()` at startup removes leftovers from crashes
(older than 1 h) — audio files never accumulate in `data/audio/`.

### State machine grew by three states
`core/state.py`: `LISTENING`, `SPEAKING`, `ERROR` (transient, cleared
automatically — `flash_error()`/`clear_error()`). The pending-sleep
mechanism (say "sleep" while thinking) now covers voice states too: a sleep
requested during LISTENING/THINKING/SPEAKING applies when the turn ends.
CSS hooks (`body[data-state=...]`) in the UI are data-attribute driven, so
new states cost zero JS.

### Settings: two layers, different lifetimes
- `.env` (config.py): keys + choices that need a restart (Gemini key/model,
  Fish key/model/voice-model, STT model size, voice master switches).
- `data/gojo_prefs.json` (prefs.py): toggles the user flips in the UI
  (voice on/off, reply-with-voice, engine choice). Tiny, validated
  (unknown keys and bad provider values are rejected), corrupted file →
  safe defaults. Engine choice applies live via `TTSRouter.set_order()`.

## Phase 4 decisions (and why)

### Why not openWakeWord (due diligence, 2026-09, checked before coding)
- Pre-trained models: alexa / hey mycroft / hey jarvis / hey rhasspy /
  weather / timers — **no "Gojo" phrase exists** in any release.
- English-only: training data is English synthetic speech; the user speaks
  Hinglish, so even a custom-trained model would underperform on their
  actual wake phrases.
- A custom "hey gojo" model needs the full training pipeline (espeak-ng on
  Windows + synthetic-data generation + hours of training), and its quality
  could not be verified without real microphone testing. We don't ship
  unverified detectors.
- Latest release is 0.6.0 (2024) — the project is moving slowly.
- faster-whisper's **tiny** model (~75 MB, int8 CPU) is already part of
  GOJO's verified stack (same wheels, same download path as the "small" STT
  model), understands Hindi + English, needs zero training for custom
  phrases, and `transcribe(..., hotwords="gojo")` biases decoding toward the
  name. Decision: whisper-based wake detection.

### Detector: sliding windows, RMS-gated, capped model
`voice/wakeword.py` — `WhisperWakeWord`: 16 kHz mono frames fill a sliding
2.0 s window (1.0 s step). Each full window is (1) RMS-gated — silent
windows skip inference entirely (near-zero idle CPU), then (2) transcribed
(tiny/int8, `beam_size=1`, `vad_filter=False`, `condition_on_previous_text=
False`, `hotwords="gojo"`, auto language), then (3) matched by
`match_wake_phrase()` — pure string logic (unit-tested): canonical phrases,
ASR variants (punctuation/case, "goyo"/"goho"/"gogo" edit distance ≤ 1,
"go jo" token split, fused "heygojo", bare "gojo"). A match resets the
buffer + a 3 s cooldown so one utterance can't fire twice. Model size is
hard-capped at tiny/base (the CPU budget is a hard constraint). `add_frame`
never raises — detection is best-effort and must not kill the listener
thread.

### Listener lifecycle = a function of (opt-in AND sleeping)
`WakeListener` owns one mic + one thread; `VoicePipeline.sync_wake()` is
the single rule: listener runs only when `always_listening` is ON **and**
state is SLEEPING. Every transition calls sync: power wake/sleep, spoken
"sleep", settings toggle, command toggle, end of a wake turn, boot. Wake
detection never runs while GOJO is awake (nothing to wake, and it keeps
"Hey Gojo" said mid-conversation from triggering a re-listen).

### Input device is explicit, never guessed
Both the recorder and the wake listener open the device from
`GOJO_MIC_DEVICE` (`.env`): empty = system default input, else an index
or a name (case-insensitive exact, then unique substring — ambiguous or
missing names fail with a readable list of available devices; there is
NO silent fallback to an arbitrary device, because virtual cable mics
like AudioRelay are NOT the system default and a silent wrong choice is
exactly the "wake word never works" bug this fixes). Resolution logic
lives in `voice/devices.py` as a pure function (device list injectable →
unit-tested without hardware). Settings → Voice surfaces the device in
use (`mic_info`) and a Test button (`test_mic`) that reports device +
frame count + peak level for ~1.2 s — level stats only, no audio is
stored or sent anywhere. The listener also counts `frames_seen` (the
"are samples actually arriving?" signal) and logs the resolved device at
each start.

### Power button semantics (Phase 4)
The UI power button is an ON/OFF switch: fully down (sleeping, no
listener) → WAKE; "wake listening" (sleeping + listener) → FULLY DOWN
(stops the listener and sets `always_listening` off — visible in
Settings); anything else → SLEEP (listener re-arms automatically only if
the setting is still on). Phase 3 bug that motivated this: "wake
listening" is `data-state="sleeping"`, so the old toggle sent `wake`
when the user pressed it expecting sleep.

### Wake turn contract: one turn per wake
On detection: SLEEPING→ACTIVE → local beep (`winsound.Beep`, 150 ms,
injectable) + "Gojo? Bolo." note (the ack is local — no network) → mic
opens (after the beep, so the beep is never recorded) → LISTENING → the
captured command runs through the SAME shared brain path (so "Gojo, sleep"
and "Gojo, always listening off" work by voice too) → reply spoken (Fish if
configured) → SLEEPING → listener re-arms (if still opted in). The turn
runs on its own worker thread — the listener thread exits immediately after
firing, because re-arming happens after the turn. No autonomous
conversation: one command per wake, by design (that's Phase 8 territory).

### Privacy (hard rule)
- `always_listening` defaults OFF and lives in `data/gojo_prefs.json`
  (validated by `prefs.py`). Nothing records unless the user opts in.
- Wake detection is fully local: no audio, frame, or transcript leaves the
  PC. Gemini/Fish are only contacted AFTER a wake, for the command/reply.
- The mic opens in exactly two situations: an explicit press-to-talk, or an
  opt-in wake listener. Nothing else.

## Phase 2 decisions (still standing)

### UI transport: pywebview bridge, NOT an HTTP server
The desktop window loads `gojo/ui/web/` and calls Python through
`window.pywebview.api` (the `GojoAPI` class). One process, zero open
ports, nothing to firewall, and the whole bridge is testable headless
(`tests/test_api.py`, `tests/test_app_wiring.py`). Voice is Python-side
(mic in, speaker out) — the UI only sends `start_talk`/`stop_talk` and
renders state; one-shot voice messages travel as a read-once `note` field
inside the existing `get_status` poll.

**When an HTTP layer WILL be added:** the Android companion (Phase 11)
or any other-device access. That becomes a thin FastAPI layer over the
same `GojoAPI`-level methods.

### Window: framed (native title bar)
Frameless + custom chrome is Phase 10 polish, done with screenshot
iteration.

### Startup state: SLEEPING
GOJO boots asleep; activation = power button (or `wake` command by text or
voice). Wake word arrives in Phase 4.

### Direct commands short-circuit the brain
"sleep" / "wake" are pattern-matched (`core/commands.py`) and answered with
canned acks — instant, free, honest. Spoken commands hit the same function.

## Extension points (what plugs in where)

| Phase | What | Where it plugs in |
|-------|------|-------------------|
| 3 — Voice | ✅ DONE: press-to-talk mic → faster-whisper → shared brain path → Fish TTS (S2.1 Pro free) → Windows-voice fallback → speakers; 6 avatar states; transcript origin tags; voice settings | `gojo/voice/` + `ui/api.py` (`start_talk`, `stop_talk`, `get_voice_settings`, `set_voice_pref`) |
| 4 — Wake word | ✅ DONE: "Hey/Hi/Hello Gojo" — local tiny-Whisper detector (openWakeWord evaluated & rejected: no Gojo models, English-only, untrainable-without-verification), opt-in always-listening (default OFF), wake → ack → same brain path → sleep + re-arm | `gojo/voice/wakeword.py` + `VoicePipeline.sync_wake()` / `_on_wake()`; prefs key `always_listening`; commands "Gojo, always listening on/off" |
| 5 — PC tools | files, apps, screenshots | new `gojo/tools/` with a tool interface; Gemini function-calling; every call passes through `gojo/security/` (Phase 12 hardens the gate) |
| 6 — Memory | remember/forget/contextual recall | `TranscriptStore` backend swaps to SQLite + local embeddings; recalled items injected into the prompt via the router — UI untouched |
| 7 — Web | search, browsing | `gojo/browser/` (DuckDuckGo + Playwright), exposed as tools to the brain |
| 8 — Agent | multi-step planning | planner lives in `gojo/core/`, executes via tools, reports via the same transcript/hero path |
| 9 — Reminders | APScheduler | scheduler triggers `state.wake()` + a UI notification; UI shows the toast hook that already exists |
| 10 — Polish | frameless window, avatar asset, talking animation tuned to real TTS | `gojo/ui/web/` only; `webview.start(icon=...)` accepts the final .ico |
| 11 — Android | USB companion | local FastAPI layer over the core + a Kotlin app; phone features only where Android permissions allow |
| 12 — Packaging | tray, autostart, installer | pystray tray icon, minimize-to-tray, Windows autostart (Run key / Task Scheduler), PyInstaller build |

## Testing rules

- Every new module ships with a runnable check in `tests/`
  (`python -m tests.test_<name>`, zero extra dependencies beyond the app's).
- The desktop bridge is guarded by a **wiring contract test**
  (`tests/test_app_wiring.py`): it drives pywebview's real
  JS→Python dispatcher (`webview.util.js_bridge_call`) against the real
  window object, and cross-checks every function `app.js` calls (parsed
  live from the file) against what pywebview actually exposes. This is
  what caught the Phase 2 "dead buttons" bug: `create_window` was called
  without `js_api`, so `window.pywebview.api` never existed in the page.
  Front-end rule that keeps this airtight: **every** bridge call goes
  through `callApi("name", ...)` so errors can never be silent.
- Anything needing a network, mic, or GPU is tested with a labeled
  stand-in: Fish's HTTP `post`, sounddevice's `stream_factory`, whisper's
  `transcribe_fn`, playback's `play_fn`, pyttsx3's `synthesize_fn`, and
  `MockProvider` (always clearly marked "not real AI" in its output).
  The real integrations are verified on the user's Windows machine.
- The live window itself is verified on the user's Windows machine via
  screenshot; what can't be tested here is said so, explicitly.

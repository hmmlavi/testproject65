"""GOJO — Personal AI Assistant (Windows)

GOJO is a personal, voice-first AI assistant that lives on your PC and talks
like Gojo Satoru would: confident, playful, occasionally roasting you, and
dead serious when it matters. Free-first, local-first, modular, no cloud
dependency.

## Where we are now

**Phase 1 — DONE:** AI brain (Gemini free tier) + terminal chat, verified.

**Phase 2 — DONE:** the real desktop application.
- [x] Desktop window (pywebview, dark glassy minimal UI, no web chrome)
- [x] UI wired to the existing Gemini brain (same brain, same personality — untouched)
- [x] Text input + responses, conversation transcript (persisted, real transcripts)
- [x] Power (wake/sleep) control; app boots SLEEPING per spec; "sleep"/"wake" commands work in chat
- [x] Status indicator (sleeping / active / thinking) + subtle state animations
- [x] Compact settings panel (brain, state, data, version, clear-data with confirm)

**Phase 3 — this build:** the voice system.
- [x] Press-to-talk: 🎙 Talk button in the chat bar. Click → listen (mic
  pulses red), speak, and GOJO auto-stops when you pause — or click again to
  stop early. **No continuous listening, ever** — the mic is only open
  during a turn you started.
- [x] Speech-to-text: faster-whisper, fully local on your PC (CPU int8,
  "small" model ≈460 MB, auto language detect — Hindi/English/Hinglish all
  work, nothing to select)
- [x] Same brain: spoken words go through the **exact same** pipeline as
  typed text (command detection → Gemini → transcript). Spoken "gojo, sleep"
  sleeps GOJO too. Text chat is completely unchanged.
- [x] Speech: Fish Audio's S2.1 Pro (free tier) with YOUR voice model ID as
  the default. If Fish is missing/rate-limited/down, it falls back to your
  Windows voice automatically (with a one-time note) — and never crashes.
- [x] Avatar states: sleeping / awake / thinking / **listening / speaking /
  error** — subtle ring motion, no giant dashboard.
- [x] Transcript integration: spoken lines are marked "· voice" and live in
  the same transcript as typed ones (no separate voice history).
- [x] Voice settings in the settings panel: on/off, reply-with-voice on/off,
  engine (auto / Fish first / local only), Fish voice, STT model.
- [x] TTS text normalization (C++ → "C plus plus", ₹ → "rupees", markdown
  stripped) applied to SPEECH only — the visible transcript stays exact.
- [x] Audio hygiene: temp voice files deleted after playback; stale files
  from crashes swept at startup; no console window while playing.
- [x] Wake-word interface created (WakeWordEngine) for Phase 4 — NOT wired to
  the mic this phase, per spec.

**Honest limits of Phase 3:**
- Mic, STT and speech need `pip install -r requirements.txt` (new voice
  packages) and a Fish key for the Fish voice (Windows-voice fallback works
  with zero keys).
- Fish Audio's free tier: no latency SLA, and per their policy requests may
  be retained for model improvement — your voice prompts leave your PC when
  you use Fish (the local fallback sends nothing anywhere).
- No wake word yet (Phase 4), no speed control (Fish supports it; added when
  more settings are worth the UI space), no custom avatar yet (yours, Phase 10).
- Memory recall, PC tools, web, reminders → later phases
- Tray icon / autostart → prepared architecturally, built in Phase 12

## Setup (one time)

```bat
cd C:\GOJO                (or wherever you cloned it)
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env    (then paste your free Gemini key into .env)
```

Free keys (both optional-but-recommended):
- Gemini (the brain): https://aistudio.google.com → **Get API key** → **Create API key**
- Fish Audio (the voice): https://fish.audio → create free account →
  API key. Your Gojo voice model ID is already pre-filled in `.env.example`
  (`FISH_AUDIO_MODEL_ID`); just add the key.

## Run

```bat
python -m gojo.app
```

- GOJO starts **sleeping** (dimmed avatar). Press the **power** button (top-right) to wake it.
- Type a message → GOJO thinks (dots + ring pulse) → replies.
- Type `gojo, sleep` → GOJO goes back down.
- **Voice:** click the 🎙 mic in the chat bar → it pulses red while
  listening → speak (Hindi/English/Hinglish, mix freely) → pause ~1.2 s and
  it stops on its own → transcript line appears (marked "· voice") → GOJO
  thinks → replies on screen **and speaks** (Fish voice if configured,
  otherwise your Windows voice) → avatar ring pulses while speaking.
  Click 🎙 again at any point to stop early.
- Top-right controls (left→right): **power** · **new chat** · **transcript** · **settings** · **quit**.
- **Voice settings** (settings panel): on/off, reply-with-voice, engine
  choice (auto / Fish first / local only), Fish voice + key status, STT model.

The old terminal chat still works too:  `python -m gojo.cli_chat`

## Troubleshooting

- **Window says "GOJO bridge offline" (full-screen message):** the UI
  couldn't reach the Python engine. Close GOJO, run `python -m gojo.app`
  again from the terminal, and read any terminal output.
- **Terminal shows `[gojo] page loaded — UI bridge ready`:** the window's
  page and bridge came up correctly; if controls still misbehave, that
  message is the marker to paste with any bug report.
- **Window doesn't open, error mentions WebView2:** install the free
  "Microsoft Edge WebView2 Runtime" from Microsoft, then run again.
- **`py` not recognized:** use `python -m venv .venv` instead.
- **Brain error when typing:** check `.env` → `GEMINI_API_KEY` is your real
  key (not the placeholder) and `GOJO_PROVIDER=gemini`.
- **"Boss, mic nahi mil raha" (no mic found):** Windows Settings → Privacy
  → Microphone → allow desktop apps; check the mic is selected under
  Settings → System → Sound → Input.
- **Speech plays in the wrong voice / Windows voice instead of Fish:**
  Settings → Voice → check "Fish key" says **ready** (if it says missing,
  put `FISH_AUDIO_API_KEY` in `.env` and restart GOJO). Settings show only
  what's configured — the key itself never appears in the UI.
- **First voice turn is slow:** faster-whisper downloads the model once
  (~460 MB) into its cache, then STT runs fully offline.
- **GOJO answered but you heard nothing:** check Windows volume/mute, and
  Settings → Voice → "Reply with voice" is on.

## Tests

```bat
python -m tests.test_router
python -m tests.test_state
python -m tests.test_commands
python -m tests.test_transcript
python -m tests.test_api
python -m tests.test_prefs
python -m tests.test_tts          (TTS abstraction, Fish contract, fallback, cleanup)
python -m tests.test_voice        (recorder, STT, full press-to-talk pipeline)
python -m tests.test_app_wiring   (bridge wiring contract, pywebview 6.x)
```

## Project layout

```
gojo/
  app.py         desktop entry point (python -m gojo.app)
  cli_chat.py    terminal chat (Phase 1, still works)
  config.py      .env loading — all settings, secrets live here only
  prefs.py       small runtime preferences (voice on/off etc.) — data/gojo_prefs.json
  transcript.py  real conversation transcripts (JSONL sessions, "via" origin marker)
  core/
    state.py     state machine: sleeping / active / thinking / listening / speaking / error
    commands.py  direct-command detection (sleep/wake) — seed of the command layer
  ai/
    base.py      AIProvider interface (brains are swappable)
    router.py    picks the brain (routing rules grow here)
    gemini.py    Gemini provider (free tier) — the brain
    mock.py      offline wiring test only — clearly labeled, NOT a real AI
    prompts.py   GOJO's personality
  voice/
    pipeline.py  press-to-talk orchestrator (mic → STT → shared brain path → TTS → speaker)
    recorder.py  microphone capture + end-of-speech detection (16 kHz, endpointing)
    stt.py       faster-whisper wrapper (lazy import, injectable for tests)
    wakeword.py  WakeWordEngine interface — Phase 4 placeholder, mic never touched
    playback.py  speakers (winsound, no console window) + audio file cleanup
    tts/
      base.py    TTSProvider interface + TTSError/TTSResult (providers are swappable)
      fish_audio.py  Fish Audio S2.1 Pro (free tier) — verified against current API docs
      windows_tts.py Windows SAPI voice via pyttsx3 (offline fallback)
      normalize.py   speech-only text cleanup (keeps the visible transcript exact)
      router.py      provider chain: Fish → local fallback, per settings
  ui/
    api.py       JS<->Python bridge — every UI capability goes through here
    web/         index.html / style.css / app.js (the front-end)
tests/           runnable checks per module (9 modules, 78 checks)
docs/
  architecture.md  the map: decisions, extension points, testing rules
data/            local data (transcripts, prefs, voice audio) — git-ignored
```

## Rules this project follows

- Real functionality only. Placeholders are labeled as placeholders.
- Free/local before any paid or cloud service.
- Secrets live only in `.env` (git-ignored), never in code or chat.
- One phase at a time; each phase is tested before the next starts.

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

**Phase 3 — DONE:** the voice system. (User-verified on the real Windows PC.)
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

**Phase 4 — this build:** real wake-word detection ("Hey Gojo").
- [x] Say **"Hey Gojo"**, **"Hi Gojo"**, **"Hello Gojo"**, **"Wake up
  Gojo"** or **"Yo Gojo"** and GOJO wakes from sleep: beep + "Gojo? Bolo."
  → listens → your command goes through the **same shared voice → brain →
  reply → TTS pipeline** → GOJO sleeps again and keeps listening.
  "Gojo, sleep" also works by voice.
- [x] **Configurable microphone device** (`.env` → `GOJO_MIC_DEVICE`):
  empty = system default input; or a device NAME/INDEX — required for
  virtual mics like **AudioRelay** (they are NOT the system default).
  Settings → Voice shows the Mic in use + a **Test** button (reports the
  device + level — nothing is recorded or stored).
- [x] **Power button is now a true ON/OFF switch** (Phase 4 semantics):
  fully down → wakes; "wake listening" → fully down (stops the listener);
  awake → sleeps (wake listening resumes if enabled).
- [x] 100% local detection: a small (tiny, ~75 MB) local Whisper model
  scans short audio windows on your CPU — **no audio ever leaves your PC**
  during wake listening. Gemini/Fish are only touched AFTER a wake.
- [x] **Always-listening is OFF by default** and user-controlled:
  Settings → Voice → "Wake word (always listening)" toggle, or say/type
  "Gojo, always listening on / off". When off, the microphone is never
  opened on its own (press-to-talk still works after power-on).
- [x] Status pill shows **sleeping / wake listening / listening /
  thinking / speaking** — no new chrome, existing style.
- [x] Wake model is capped at "tiny" (CPU budget) and silent windows skip
  inference entirely — near-zero idle CPU.
- [x] openWakeWord was evaluated and rejected (documented in
  docs/architecture.md): its pre-trained models don't include any "Gojo"
  phrase and it's English-only; a custom model would need a heavy training
  pipeline we can't verify without your microphone.

**Honest limits (Phase 3+4):**
- Mic, STT and speech need `pip install -r requirements.txt` (voice
  packages) and a Fish key for the Fish voice (Windows-voice fallback works
  with zero keys).
- First wake-listening session downloads the ~75 MB "tiny" model once,
  then runs offline.
- Wake-word detection quality (real speech, your room, your mic) can only
  be judged on your PC — everything else is tested (see tests).
- Fish Audio's free tier: no latency SLA, and per their policy requests may
  be retained for model improvement — your voice prompts leave your PC when
  you use Fish (the local fallback and the entire wake-word stage send
  nothing anywhere).
- No speed control, no custom avatar yet (yours, Phase 10).
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
- **Wake word (opt-in):** Settings → Voice → tick **"Wake word (always
  listening)"** (or say "Gojo, always listening on" while awake). The pill
  now reads **wake listening**. Even while sleeping, say **"Hey Gojo"** →
  beep + "Gojo? Bolo." → speak your command → GOJO replies and speaks →
  sleeps again. Say "Gojo, always listening off" (or untick) to stop; the
  mic then only opens via power + 🎙 as before.
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
- **Voice stops responding for any reason (no transcript / no wake):**
  run the voice doctor on the PC (close the GOJO window first):
  ```bat
  python -m gojo.voice.doctor
  ```
  It walks the exact runtime path — device → stream rate → capture →
  peak/RMS → resample → Whisper load → transcript → brain routing →
  wake-listener probe — and prints a numbered report (`[1]`…`[10]`,
  `[W1]`…`[W5]`). Send the whole output. Nothing is recorded, stored, or
  sent anywhere; no keys are read.
  If you get audio but an EMPTY transcript, run the deep-dive — it
  transcribes the same sample 4 ways (app config / old VAD-on /
  `language=hi` / `language=en`), prints detected language + probability,
  segment count, per-segment text/probability, and what the wake matcher
  makes of the actual recognized text:
  ```bat
  python -m gojo.voice.doctor --diagnose
  ```
  If one explicit language clearly wins, pin it in `.env` with
  `GOJO_STT_LANGUAGE=hi` (or `en`). Also: Settings → Voice → **Test**
  button now proves the mic is non-silent (shows samples, peak, RMS and
  says `MIC SILENT` if no audio is arriving — e.g. AudioRelay left in
  Playback mode).
- **Saying "Hey Gojo" does nothing:** (1) Settings → Voice → "Wake word"
  must be ticked and the pill must say **wake listening** (not just
  sleeping); (2) **check the Mic row + Test button** — GOJO listens to
  the device from `.env` (`GOJO_MIC_DEVICE`, empty = system default).
  If your phone mic comes in through **AudioRelay**, that virtual device
  is NOT the default input: list devices with
  `python -c "import sounddevice as sd; print(sd.query_devices())"` and
  set `GOJO_MIC_DEVICE` to its name or index, then restart. (Sample rate
  is automatic: GOJO opens the stream at the device's native rate — e.g.
  48 kHz on AudioRelay — and converts to the 16 kHz the speech models
  need; if you ever see `Invalid sample rate [PaErrorCode -9997]` you're
  on a pre-fix build — `git pull`.) (3) mic
  permission — Windows Settings → Privacy → Microphone → allow desktop
  apps; (4) say it clearly, ~0.5–1 m from the mic, in a reasonably quiet
  room; (5) first time adds ~2–3 s detection latency (2 s window + local
  inference) and downloads the ~75 MB model once; (6) try "Hello Gojo" /
  "Wake up Gojo" / "Yo Gojo" — all five phrases are supported and ASR
  varies by speaker.
- **GOJO wakes when it shouldn't:** the detector is local and fuzzy by
  design (it tolerates mis-recognized "gojo"); the false-wake cost is one
  "I didn't catch that" and it goes back to listening. If it's a nuisance
  in a loud room, keep wake word off and use power + 🎙.

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
python -m tests.test_wakeword     (wake matching, listener lifecycle, wake flow)
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
    wakeword.py  LOCAL wake word (Phase 4): tiny-Whisper windows + fuzzy
                 "Hey/Hi/Hello Gojo" matcher + mic listener (opt-in only)
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
tests/           runnable checks per module (10 modules, 100 checks)
docs/
  architecture.md  the map: decisions, extension points, testing rules
data/            local data (transcripts, prefs, voice audio) — git-ignored
```

## Rules this project follows

- Real functionality only. Placeholders are labeled as placeholders.
- Free/local before any paid or cloud service.
- Secrets live only in `.env` (git-ignored), never in code or chat.
- One phase at a time; each phase is tested before the next starts.

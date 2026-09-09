"""GOJO — Personal AI Assistant (Windows)

GOJO is a personal, voice-first AI assistant that lives on your PC and talks
like Gojo Satoru would: confident, playful, occasionally roasting you, and
dead serious when it matters. Free-first, local-first, modular, no cloud
dependency.

## Where we are now

**Phase 1 — DONE:** AI brain (Gemini free tier) + terminal chat, verified.

**Phase 2 — this build:** the real desktop application.
- [x] Desktop window (pywebview, dark glassy minimal UI, no web chrome)
- [x] UI wired to the existing Gemini brain (same brain, same personality — untouched)
- [x] Text input + responses, conversation transcript (persisted, real transcripts)
- [x] Power (wake/sleep) control; app boots SLEEPING per spec; "sleep"/"wake" commands work in chat
- [x] Status indicator (sleeping / active / thinking) + subtle state animations
- [x] Compact settings panel (brain, state, data, version, clear-data with confirm)
- [x] Architecture hooks for voice, memory, tools, agent (see docs/architecture.md)

**Not in Phase 2 (honest list — nothing here fakes them):**
- Voice (mic/TTS/wake word) → Phase 3
- Memory recall, PC tools, web, reminders → later phases
- Your avatar asset → provided by you, integrated in Phase 10 (a neutral
  "G" monogram placeholder is shown until then — no fabricated avatar)
- Tray icon / autostart → prepared architecturally, built in Phase 12

## Setup (one time)

```bat
cd C:\GOJO                (or wherever you cloned it)
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env    (then paste your free Gemini key into .env)
```

Free key: https://aistudio.google.com → **Get API key** → **Create API key**

## Run

```bat
python -m gojo.app
```

- GOJO starts **sleeping** (dimmed avatar). Press the **power** button (top-right) to wake it.
- Type a message → GOJO thinks (dots + ring pulse) → replies.
- Type `gojo, sleep` → GOJO goes back down.
- Top-right controls (left→right): **power** · **new chat** · **transcript** · **settings** · **quit**.

The old terminal chat still works too:  `python -m gojo.cli_chat`

## Troubleshooting

- **Window doesn't open, error mentions WebView2:** install the free
  "Microsoft Edge WebView2 Runtime" from Microsoft, then run again.
- **`py` not recognized:** use `python -m venv .venv` instead.
- **Brain error when typing:** check `.env` → `GEMINI_API_KEY` is your real
  key (not the placeholder) and `GOJO_PROVIDER=gemini`.

## Tests

```bat
python -m tests.test_router
python -m tests.test_state
python -m tests.test_commands
python -m tests.test_transcript
python -m tests.test_api
```

## Project layout

```
gojo/
  app.py         desktop entry point (python -m gojo.app)
  cli_chat.py    terminal chat (Phase 1, still works)
  config.py      .env loading — all settings, secrets live here only
  transcript.py  real conversation transcripts (JSONL sessions)
  core/
    state.py     state machine: sleeping / active / thinking (+voice states later)
    commands.py  direct-command detection (sleep/wake) — seed of the command layer
  ai/
    base.py      AIProvider interface (brains are swappable)
    router.py    picks the brain (routing rules grow here)
    gemini.py    Gemini provider (free tier) — the brain
    mock.py      offline wiring test only — clearly labeled, NOT a real AI
    prompts.py   GOJO's personality
  ui/
    api.py       JS<->Python bridge — every UI capability goes through here
    web/         index.html / style.css / app.js (the front-end)
tests/           runnable checks per module
docs/
  architecture.md  the map: decisions, extension points, testing rules
data/            local data (transcripts) — git-ignored
```

## Rules this project follows

- Real functionality only. Placeholders are labeled as placeholders.
- Free/local before any paid or cloud service.
- Secrets live only in `.env` (git-ignored), never in code or chat.
- One phase at a time; each phase is tested before the next starts.

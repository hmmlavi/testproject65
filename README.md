"""GOJO — Personal AI Assistant (Windows)

GOJO is a personal, voice-first AI assistant that lives on your PC and talks
like Gojo Satoru would: confident, playful, occasionally roasting you, and
dead serious when it matters. Free-first, local-first, modular, no cloud
dependency.

## Where we are now

**Phase 1 (in progress): AI brain + text chat**
- [x] Modular AI layer — brains are swappable via config, nothing hard-coded to one provider
- [x] GOJO personality (playful, Hinglish, zero corporate tone, never fakes actions)
- [x] Terminal chat — a developer front-end to prove the brain is real,
      before we build the desktop window

**Next steps (in order):**
1. Desktop app window (glassy UI, chat) — next
2. Voice in/out (faster-whisper STT + Piper TTS, both local & free)
3. Wake word ("Hey Gojo" / "Wake up Gojo") + sleep state
4. Personality polish
5. PC tools (files, apps, VS Code, screenshots) behind a security layer
6. Memory + conversation history (SQLite)
7. Internet (search, web, browser)
8. Agent mode (multi-step tasks)
9. Reminders/scheduling
10. Avatar + UI polish
11. Android companion
12. Security hardening + packaging

## Setup (one time, Windows)

```bat
cd C:\path\where\you\cloned\GOJO
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
:: now open .env in any editor and paste your free Gemini key
```

Free key: https://aistudio.google.com → **Get API key** → **Create API key**

## Run the terminal chat

```bat
python -m gojo.cli_chat
```

Then just talk. Try: `gojo, tu kaun hai?` — type `/help` for commands.

### No-key wiring check (optional, NOT a real AI)

```bat
set GOJO_PROVIDER=mock
python -m gojo.cli_chat
```

Echoes your text with a MOCK label — only proves the install works.
Remove it with `set GOJO_PROVIDER=gemini` (or close/reopen the terminal).

## Project layout

```
gojo/
  config.py      .env loading, all settings in one place
  cli_chat.py    terminal chat (Phase 1 dev front-end)
  ai/
    base.py      AIProvider interface (what makes brains swappable)
    router.py    picks the brain from config (routing rules live here)
    gemini.py    Google Gemini provider (free tier) — default brain
    mock.py      offline wiring test only — clearly labeled, not real AI
    prompts.py   GOJO's personality (system prompt)
tests/           runnable checks:  python -m tests.test_router
data/            local data (memory DB etc.) — created later, git-ignored
```

## Rules this project follows

- Real functionality only. Placeholders are labeled as placeholders.
- Free/local before any paid or cloud service.
- Secrets live only in `.env` (git-ignored), never in code or chat.
- One phase at a time; each phase is tested before the next starts.

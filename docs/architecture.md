# GOJO — Architecture Map

Updated: Phase 2 (desktop app). Read this before adding anything new.

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
  Desktop UI ──────▶│  core/state   (sleeping / active / thinking)
  (pywebview window)│  core/commands (sleep / wake detection)
   via ui/api.py ──▶│  transcript   (JSONL sessions → SQLite in Ph.6)
                    │  config       (.env — secrets live here only)
                    └────────────────────────────────────────────┘
```

## Phase 2 decisions (and why)

### UI transport: pywebview bridge, NOT an HTTP server
The desktop window loads `gojo/ui/web/` and calls Python through
`window.pywebview.api` (the `GojoAPI` class). One process, zero open
ports, nothing to firewall, and the whole bridge is testable headless
(`tests/test_api.py`). Voice in Phase 3 is Python-side anyway (mic in,
speaker out) — the UI only shows state.

**When an HTTP layer WILL be added:** the Android companion (Phase 11)
or any other-device access. That becomes a thin FastAPI layer over the
same `GojoAPI`-level methods. Nothing in Phase 2 assumes a transport.

### Window: framed (native title bar) in Phase 2
pywebview 6.x supports `frameless=True` + `easy_drag`, but a frameless
window with no tested fallback is how "premium app" turns into
"stuck window" on the user's first run. Frameless + custom window
chrome is part of Phase 10 polish, done with screenshot iteration.

### Startup state: SLEEPING
Per spec, GOJO starts with the PC in a sleep state and only activates on
request. Phase 2 activation = power button (or typing a wake command);
Phase 3 adds "Wake up Gojo" by voice.

### Direct commands short-circuit the brain
"sleep" / "wake" are pattern-matched (`core/commands.py`) and answered
with canned acks — instant, free, honest (they don't need the brain).
This module is the seed of the command layer: voice transcripts will go
through the same function in Phase 3.

## Extension points (what plugs in where)

| Phase | What | Where it plugs in |
|-------|------|-------------------|
| 3 — Voice | mic capture + VAD + faster-whisper STT | new `gojo/voice/` package; detected speech feeds the same `GojoAPI.chat()` path; new states `LISTENING`, `SPEAKING` added to the same enum |
| 3 — Wake word | "Hey Gojo" / "Wake up Gojo" | `gojo/voice/wake.py`; on detection calls `state.wake()` (button and wake word share one path) |
| 3 — TTS | Piper (local) / Fish Audio (optional) | `gojo/voice/tts.py`; `SPEAKING` state drives the existing `.speaking` CSS hook |
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
  (`python -m tests.test_<name>`, zero extra dependencies).
- The desktop bridge is guarded by a **wiring contract test**
  (`tests/test_app_wiring.py`): it drives pywebview's real
  JS→Python dispatcher (`webview.util.js_bridge_call`) against the real
  window object, and cross-checks every function `app.js` calls (parsed
  live from the file) against what pywebview actually exposes. This is
  what caught the Phase 2 "dead buttons" bug: `create_window` was called
  without `js_api`, so `window.pywebview.api` never existed in the page.
  Front-end rule that keeps this airtight: **every** bridge call goes
  through `callApi("name", ...)` so errors can never be silent.
- Anything needing a network, mic, or window is tested with a labeled
  stand-in (MockProvider) — and the stand-in is always clearly marked
  "not real AI" in its output.
- The live window itself is verified on the user's Windows machine via
  screenshot; what can't be tested here is said so, explicitly.

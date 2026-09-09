"""GOJO desktop application — entry point.

Run:
    python -m gojo.app

Starts the local desktop window (pywebview) wired to the existing AI brain.
No server, no open ports: the UI talks to Python through pywebview's
built-in JS bridge (gojo/ui/api.py). If a later phase needs a real local
HTTP API (Android companion, etc.), it is added on top of the same core.

Per spec, GOJO boots in the SLEEPING state — press power (or, in Phase 3,
"Wake up Gojo") to activate it.
"""
from __future__ import annotations

import sys

import webview

from . import __version__
from .ai.base import AIProvider
from .ai.router import build_router
from .config import Settings, load_settings
from .core.state import AppState, StateManager
from .transcript import TranscriptStore
from .ui import WEB_DIR
from .ui.api import GojoAPI


def create_app(settings: Settings | None = None):
    """Build the full app (window + bridge) WITHOUT starting the event loop.

    CRITICAL: `js_api=api` is what makes `window.pywebview.api` exist inside
    the page. Without it the front-end has no bridge and every button is
    dead — this exact regression is guarded by tests/test_app_wiring.py.

    Returns (window, api). main() starts the loop; tests can build the app
    headless and verify the real pywebview wiring.
    """
    if settings is None:
        settings = load_settings()

    # The brain is optional at startup: if the key is missing we still open
    # the window, and the chat surfaces the friendly setup instructions.
    brain: AIProvider | None = None
    brain_error: str | None = None
    try:
        brain = build_router(settings)
    except (RuntimeError, ValueError) as exc:
        brain_error = str(exc)

    state = StateManager(initial=AppState.SLEEPING)
    transcript = TranscriptStore(settings.data_dir)
    api = GojoAPI(
        settings=settings,
        brain=brain,
        state=state,
        transcript=transcript,
        brain_error=brain_error,
        version=__version__,
    )

    # Resume the previous session's context so the AI history matches the
    # transcript the user sees on startup.
    for item in transcript.load():
        role = item.get("role")
        if role in ("user", "model"):
            api.note_history(role, item.get("text", ""))

    window = webview.create_window(
        title="GOJO",
        url=str(WEB_DIR / "index.html"),
        js_api=api,  # <-- the bridge. Exposed to the page as window.pywebview.api
        width=1060,
        height=720,
        min_size=(900, 620),
        background_color="#0b0e13",
    )
    api.attach_window(window)

    def _on_loaded(*_args) -> None:
        print("[gojo] page loaded — UI bridge ready")

    window.events.loaded += _on_loaded

    return window, api


def main() -> int:
    create_app()
    webview.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())

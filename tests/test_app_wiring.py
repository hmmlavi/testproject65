"""Wiring contract tests for the desktop bridge (tested against pywebview 6.2.1).

This is the guard for the "buttons do nothing" regression: it uses
pywebview's REAL internals (not a mock of pywebview) to verify that:

1. create_app() passes `js_api` to create_window — i.e. the page will
   actually receive `window.pywebview.api` (without it, every button is
   dead and the only symptom was a tiny "bridge offline" pill).
2. Every function the front-end calls (parsed LIVE from app.js) resolves
   through pywebview's real JS->Python dispatcher
   (`webview.util.js_bridge_call`) against the real window object.
3. A dispatched call actually changes GOJO state (side effects, not just
   "the call was accepted").
4. The window URL points at an existing page that loads the bridge JS.

Run:  python -m tests.test_app_wiring
"""
from __future__ import annotations

import logging
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import webview
import webview.util

from gojo import app as gojo_app
from gojo.config import Settings
from gojo.core.state import AppState
from gojo.ui import WEB_DIR
from gojo.ui.api import GojoAPI

APP_JS = WEB_DIR / "app.js"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _settings(tmp: Path) -> Settings:
    return Settings(
        gemini_api_key=None,
        gemini_model="mock-model",
        provider="mock",  # labeled offline stand-in — no network in tests
        data_dir=tmp / "data",
        log_level="INFO",
    )


def _build(tmp: Path):
    window, api = gojo_app.create_app(settings=_settings(tmp))
    return window, api


def _js_called_names() -> set[str]:
    """All bridge functions the front-end calls. app.js routes every call
    through callApi("name", ...) by rule — the guard below enforces that."""
    js = APP_JS.read_text(encoding="utf-8")
    direct = re.findall(r"(?<![\w.])api\.(\w+)\(", js)
    assert not direct, (
        f"app.js must route every bridge call through callApi() so errors "
        f"cannot be silent. Direct calls found: {sorted(set(direct))}"
    )
    return set(re.findall(r'callApi\(\s*"(\w+)"', js))


def _wait_until(cond, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


# ---------------------------------------------------------------------
# the real dispatcher
# ---------------------------------------------------------------------

def _quiet_hook(*_exc_info) -> None:
    """Headless: the dispatcher's return path (window.evaluate_js) needs a
    live renderer and raises by design. Silence it (both hooks: thread
    exceptions go through threading.excepthook, not sys.excepthook)."""


def _dispatch(window, name: str, args: list) -> bool:
    """Send a call through pywebview's REAL JS->Python dispatcher
    (webview.util.js_bridge_call) and return True if the function resolved.

    'Function X() does not exist' is logged synchronously by the dispatcher
    when the name cannot be found on the js_api object — exactly the failure
    the renderer would hit.
    """
    handler = _Capture()
    logger = logging.getLogger("pywebview")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    real_sys, real_thread = sys.excepthook, threading.excepthook
    try:
        if not window.events._pywebviewready.is_set():
            window.events._pywebviewready.set()  # fail fast, no 20s wait headless
        sys.excepthook = _quiet_hook
        threading.excepthook = _quiet_hook
        webview.util.js_bridge_call(window, name, list(args), f"test-{name}")
        resolved = not any("does not exist" in r.getMessage() for r in handler.records)
        time.sleep(0.3)  # let the worker thread finish its (expected) fail-fast
        return resolved
    finally:
        sys.excepthook, threading.excepthook = real_sys, real_thread
        logger.removeHandler(handler)


def test_js_api_attached_to_window() -> None:
    """THE regression: create_window must receive js_api=api, otherwise
    window.pywebview.api is undefined in the page and all buttons are dead."""
    with tempfile.TemporaryDirectory() as tmp:
        window, api = _build(Path(tmp))
        assert window._js_api is api, (
            "create_window() was called WITHOUT js_api — the page will have "
            "no window.pywebview.api and every control will be dead."
        )
        assert api._window is window
    print("PASS test_js_api_attached_to_window")


def test_page_loads_bridge_js() -> None:
    """The window must point at a real page that loads the bridge script."""
    with tempfile.TemporaryDirectory() as tmp:
        window, _ = _build(Path(tmp))
        url = Path(window.original_url)
        assert url.exists(), f"window url does not exist: {url}"
        html = url.read_text(encoding="utf-8")
        assert 'src="app.js"' in html, "index.html does not load app.js"
        assert (url.parent / "app.js").exists()
        assert (url.parent / "style.css").exists()
    print("PASS test_page_loads_bridge_js")


def test_every_js_call_resolves_in_dispatcher() -> None:
    """Every bridge function app.js calls must resolve through pywebview's
    real dispatcher (or, for window controls that need a live GUI, be a
    callable on the api object)."""
    names = _js_called_names()
    assert {"wake", "sleep", "chat", "get_status"} <= names, (
        f"app.js stopped calling core bridge functions? parsed: {sorted(names)}"
    )
    live = {n for n in names if n not in ("close_app", "minimize")}
    with tempfile.TemporaryDirectory() as tmp:
        window, _ = _build(Path(tmp))
        bad = [n for n in sorted(live) if not _dispatch(window, n, _ARGS_FOR.get(n, []))]
        assert not bad, f"pywebview dispatcher cannot resolve these bridge names: {bad}"
        # window controls: resolvable on the api (they need a live GUI to run)
        for n in sorted(names - live):
            assert callable(getattr(GojoAPI, n, None)), f"{n} is not a GojoAPI method"
    print("PASS test_every_js_call_resolves_in_dispatcher")


def test_dispatch_changes_gojo_state() -> None:
    """Dispatched calls must produce REAL side effects (state + transcript),
    verified through the real dispatcher thread — not by calling GojoAPI
    directly."""
    with tempfile.TemporaryDirectory() as tmp:
        window, api = _build(Path(tmp))
        # 1) power: wake
        _dispatch(window, "wake", [])
        assert _wait_until(lambda: api._state.state is AppState.ACTIVE), "wake did not change state"

        # 2) chat through the dispatcher -> real transcript entries
        probe = "wiring probe ping"
        _dispatch(window, "chat", [probe])
        assert _wait_until(
            lambda: any(i.get("text") == probe for i in api._transcript.load())
        ), "dispatched chat did not reach the transcript"

        # 3) sleep
        _dispatch(window, "sleep", [])
        assert _wait_until(lambda: api._state.state is AppState.SLEEPING), "sleep did not change state"

        # 4) new chat resets
        _dispatch(window, "new_chat", [])
        assert _wait_until(lambda: api._transcript.load() == []), "new_chat did not reset transcript"
    print("PASS test_dispatch_changes_gojo_state")


# args for side-effect-free dispatch checks (state changes are fine in tests)
_ARGS_FOR = {
    "get_status": [],
    "wake": [],
    "sleep": [],
    "new_chat": [],
    "get_transcript": [],
    "get_settings": [],
    "clear_data": [],
    "chat": ["wiring probe (resolve check)"],
}


def test_js_names_are_public_api_methods() -> None:
    names = _js_called_names()
    for name in sorted(names):
        attr = getattr(GojoAPI, name, None)
        assert callable(attr), f"app.js calls bridge name {name!r} but GojoAPI has no such method"
    print("PASS test_js_names_are_public_api_methods")


def main() -> int:
    tests = [
        test_js_api_attached_to_window,
        test_page_loads_bridge_js,
        test_every_js_call_resolves_in_dispatcher,
        test_dispatch_changes_gojo_state,
        test_js_names_are_public_api_methods,
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

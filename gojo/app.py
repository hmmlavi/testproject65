"""GOJO desktop application — entry point.

Run:
    python -m gojo.app

Starts the local desktop window (pywebview) wired to the existing AI brain
and the Phase 3 voice pipeline. No server, no open ports: the UI talks to
Python through pywebview's built-in JS bridge (gojo/ui/api.py). If a later
phase needs a real local HTTP API (Android companion, etc.), it is added on
top of the same core.

Per spec, GOJO boots in the SLEEPING state — press power (or, in Phase 4,
"Wake up Gojo") to activate it. Voice is press-to-talk: the mic is only
ever open while the user is talking.
"""
from __future__ import annotations

import sys

import webview

from . import __version__
from .ai.base import AIProvider
from .ai.router import build_router
from .config import Settings, load_settings
from .core.state import AppState, StateManager
from .prefs import Prefs
from .transcript import TranscriptStore
from .ui import WEB_DIR
from .ui.api import GojoAPI
from .voice.pipeline import VoicePipeline
from .voice.playback import AudioPlayer
from .voice.recorder import VoiceRecorder
from .voice.stt import WhisperSTT
from .voice.tts.fish_audio import FishAudioTTS
from .voice.tts.router import TTSRouter
from .voice.tts.windows_tts import WindowsTTS


def create_app(settings: Settings | None = None):
    """Build the full app (window + bridge + voice) WITHOUT starting the
    event loop.

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
    prefs = Prefs(
        settings.data_dir / "gojo_prefs.json",
        defaults={
            "voice_enabled": settings.voice_enabled,
            "tts_autoplay": settings.tts_autoplay,
            "tts_provider": settings.tts_provider,
        },
    )
    api = GojoAPI(
        settings=settings,
        brain=brain,
        state=state,
        transcript=transcript,
        brain_error=brain_error,
        version=__version__,
        prefs=prefs,
    )

    # Resume the previous session's context so the AI history matches the
    # transcript the user sees on startup.
    for item in transcript.load():
        role = item.get("role")
        if role in ("user", "model"):
            api.note_history(role, item.get("text", ""))

    # Voice (Phase 3): recorder -> whisper STT -> shared brain path ->
    # TTS (Fish, local Windows fallback) -> speakers.
    # Heavy dependencies (sounddevice, faster-whisper, pyttsx3) are imported
    # lazily inside first use, so building the app stays light and fast.
    player = AudioPlayer()
    AudioPlayer.sweep_stale(settings.data_dir / "audio")
    pipeline = VoicePipeline(
        state=state,
        # device_spec: GOJO_MIC_DEVICE from .env ('' = system default input).
        # Virtual mics (e.g. AudioRelay) are NOT the default input, so a
        # phone-mic setup must name the device explicitly — the Settings
        # panel shows which device GOJO actually uses + a Test button.
        recorder=VoiceRecorder(device_spec=settings.mic_device),
        stt=WhisperSTT(settings.stt_model_size),
        tts=TTSRouter(
            [
                FishAudioTTS(
                    api_key=settings.fish_api_key,
                    model_id=settings.fish_model_id,
                    model=settings.fish_model,
                ),
                WindowsTTS(),
            ],
            order=prefs.get("tts_provider"),
        ),
        player=player,
        on_text=api._run_turn,
        prefs=prefs,
        on_note=api._set_note,
        audio_dir=settings.data_dir / "audio",
        mic_device=settings.mic_device,
    )
    api.attach_voice(pipeline)
    # Phase 4: if the user previously opted into always-listening, resume
    # the LOCAL wake listener at boot (GOJO boots SLEEPING). Off by default.
    pipeline.sync_wake()

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

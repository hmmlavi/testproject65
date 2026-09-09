"""Wake-word engine interface (Phase 4 slot — NOT active in Phase 3).

Phase 3 privacy default: the microphone opens ONLY while the user is
press-to-talking. Nothing here listens continuously.

Phase 4 will plug a local wake-word engine (openWakeWord / Porcupine or a
similar free engine) into this interface:

    engine = PorcupineWakeWordEngine(keywords=("hey gojo", "wake up gojo"))
    engine.start(on_trigger=lambda: state.wake())
    # on_trigger fires from a worker thread when a keyword is detected
    engine.stop()

Adding "always listening on/off" later means: user opts in explicitly in
settings -> engine.start() / engine.stop(). It never turns on by itself.
"""
from __future__ import annotations

from typing import Callable


class WakeWordEngine:
    """Interface every wake-word engine must implement."""

    name: str = "base"

    def __init__(self, keywords: tuple[str, ...], on_trigger: Callable[[], None]) -> None:
        self.keywords = keywords
        self.on_trigger = on_trigger

    def available(self) -> bool:
        """Phase 3: no engine is installed, so this is False everywhere."""
        return False

    def start(self) -> None:
        """Start continuous keyword detection (background thread)."""
        raise NotImplementedError("Wake-word engine arrives in Phase 4")

    def stop(self) -> None:
        """Stop detection and release the microphone."""
        raise NotImplementedError("Wake-word engine arrives in Phase 4")


def build_wakeword_engine(
    keywords: tuple[str, ...], on_trigger: Callable[[], None]
) -> WakeWordEngine | None:
    """Phase 3: always None — no engine installed.

    Phase 4: inspects installed engines (Porcupine / openWakeWord) and
    returns the first available one. The rest of GOJO already codes
    against this function, so Phase 4 changes nothing outside voice/.
    """
    return None

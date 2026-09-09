"""OFFLINE MOCK provider — NOT a real AI.

Exists ONLY so you can verify the installation and code wiring work without
an API key. It just echoes your text back with a MOCK label. Nothing here
pretends to be intelligent.

Use with:   set GOJO_PROVIDER=mock   then   python -m gojo.cli_chat
For real GOJO, remove GOJO_PROVIDER=mock (the default is "gemini").
"""
from __future__ import annotations

from .base import AIProvider, ChatMessage


class MockProvider(AIProvider):
    name = "mock"

    def chat(self, messages: list[ChatMessage], system_prompt: str) -> str:
        last = messages[-1].text if messages else ""
        return (
            f"[MOCK — offline wiring test only, NOT a real AI] I got your text: {last!r}.\n"
            "Your Python + GOJO setup is working end to end. To talk to the real GOJO: "
            "put GEMINI_API_KEY in .env and remove GOJO_PROVIDER=mock."
        )

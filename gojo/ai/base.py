"""AI layer interface.

Every "brain" GOJO can use must implement this interface. This is what makes
the AI model layer replaceable: the rest of GOJO only ever talks to
`AIProvider`, never directly to Gemini/Ollama/anything else. Later, the
router can switch between providers (reasoning model for hard tasks,
local model for private tasks, etc.) without touching any other code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatMessage:
    """One message in the conversation."""

    role: str  # "user" or "model"
    text: str


class AIProvider:
    """Interface that all AI providers (brains) implement."""

    name: str = "base"

    def chat(self, messages: list[ChatMessage], system_prompt: str) -> str:
        """Send the conversation history + system prompt, return reply text."""
        raise NotImplementedError

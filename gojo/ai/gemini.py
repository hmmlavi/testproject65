"""Google Gemini provider (free tier) — GOJO's default brain.

Uses Google's official `google-genai` SDK. The API key is read from the
local `.env` file — it is never hard-coded and never logged.
"""
from __future__ import annotations

from google import genai
from google.genai import types

from .base import AIProvider, ChatMessage


class GeminiProvider(AIProvider):
    name = "gemini"

    def __init__(self, api_key: str | None, model: str = "gemini-2.5-flash") -> None:
        if not api_key or api_key.strip().lower().startswith("paste-"):
            raise RuntimeError(
                "GEMINI_API_KEY is not set.\n"
                "1) Copy .env.example to .env  (in cmd:  copy .env.example .env)\n"
                "2) Get a FREE key at https://aistudio.google.com  ->  Get API key\n"
                "3) Paste it into .env on the GEMINI_API_KEY line."
            )
        self._client = genai.Client(api_key=api_key.strip())
        self._model = model

    def chat(self, messages: list[ChatMessage], system_prompt: str) -> str:
        contents = [
            types.Content(role=m.role, parts=[types.Part.from_text(text=m.text)])
            for m in messages
        ]
        config = types.GenerateContentConfig(system_instruction=system_prompt)
        response = self._client.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        text = response.text
        if not text:
            # Rare: safety filter blocked the output or empty response.
            return "(My reply got blocked by a safety filter — try rephrasing, boss.)"
        return text.strip()

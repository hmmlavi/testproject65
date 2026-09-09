"""Fish Audio TTS provider — GOJO's primary voice (Phase 3).

API contract verified against docs.fish.audio (Sept 2026):
    POST https://api.fish.audio/v1/tts
    Headers:
        Authorization: Bearer <FISH_AUDIO_API_KEY>
        model: <model string — "s2.1-pro-free" for the free tier>
        Content-Type: application/json
    Body:
        {"text": ..., "reference_id": <voice model id>, "format": "mp3"}
    200  -> raw MP3 bytes (written straight to a temp file)
    !200 -> JSON error (status + detail surfaced as TTSError)

Free tier notes (official docs): fair-use, no SLA, requests may be retained
for model improvement — fine for a personal assistant, worth knowing.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .base import TTSError, TTSProvider

logger = logging.getLogger("gojo.voice.tts.fish")

ENDPOINT = "https://api.fish.audio/v1/tts"
MIN_AUDIO_BYTES = 1024  # a "200" that returns less than this is a bad payload


class FishAudioTTS(TTSProvider):
    name = "fish"

    def __init__(
        self,
        api_key: str | None,
        model_id: str,
        model: str = "s2.1-pro-free",
        post=None,
    ) -> None:
        self._key = (api_key or "").strip()
        self._model_id = (model_id or "").strip()
        self._model = (model or "s2.1-pro-free").strip()
        # injectable transport: (url, headers, body_bytes) -> (status, bytes)
        self._post = post if post is not None else self._default_post

    @property
    def available(self) -> bool:
        """Available only with a real key (not the .env placeholder) + voice id."""
        if not self._key or self._key.lower().startswith("paste-"):
            return False
        return bool(self._model_id)

    def synthesize(self, text: str, out_dir: Path) -> Path:
        if not self.available:
            raise TTSError(
                "Fish Audio is not configured (missing API key or voice id in .env)",
                provider=self.name,
                retryable=False,
            )
        body = json.dumps(
            {"text": text, "reference_id": self._model_id, "format": "mp3"}
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "model": self._model,
        }
        try:
            status, payload = self._post(ENDPOINT, headers, body)
        except Exception as exc:  # noqa: BLE001 — network failure of any kind
            raise TTSError(f"Fish Audio unreachable: {exc}", provider=self.name) from exc

        if status != 200:
            detail = payload[:200].decode("utf-8", errors="replace").strip()
            raise TTSError(
                f"Fish Audio HTTP {status}: {detail}".rstrip(": "),
                provider=self.name,
            )
        if len(payload) < MIN_AUDIO_BYTES:
            raise TTSError(
                f"Fish Audio returned a bad payload ({len(payload)} bytes)",
                provider=self.name,
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"gojo-fish-{int(time.time() * 1000)}-{os.urandom(2).hex()}.mp3"
        path.write_bytes(payload)
        return path

    @staticmethod
    def _default_post(url: str, headers: dict[str, str], data: bytes) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        # URLError / timeout / DNS -> propagates as Exception -> TTSError(retryable)

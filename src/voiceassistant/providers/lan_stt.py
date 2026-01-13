"""LAN OpenAI-compatible STT provider."""

from __future__ import annotations

from typing import Optional

import requests

from voiceassistant.logging_config import get_logger
from voiceassistant.providers.base import ProviderError
from voiceassistant.providers.http import HttpProvider, STTResponse
from voiceassistant.providers.stt import STTProvider

logger = get_logger(__name__)


class LanHttpSTTProvider(HttpProvider, STTProvider):
    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout_s: float,
        api_key: Optional[str],
    ) -> None:
        super().__init__(name, endpoint, timeout_s, api_key)

    def transcribe(self, wav_bytes: bytes) -> STTResponse:
        if not isinstance(wav_bytes, (bytes, bytearray)):
            raise ProviderError("STT input must be WAV bytes")
        if len(wav_bytes) < 12 or wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            raise ProviderError("STT input must be WAV bytes")

        files = {"file": ("audio.wav", bytes(wav_bytes), "audio/wav")}
        try:
            resp = requests.post(
                self.endpoint,
                headers=self._headers(),
                files=files,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code != 200:
            snippet = resp.text.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = f"{snippet[:200]}..."
            message = f"HTTP {resp.status_code}"
            if snippet:
                message = f"{message}: {snippet}"
            raise ProviderError(message)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ProviderError("Invalid JSON in STT response") from exc
        if "text" not in payload:
            raise ProviderError("STT response missing required 'text' field")
        text = payload.get("text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ProviderError("Invalid 'text' in STT response")
        logger.debug("STT transcript length=%d", len(text))
        return STTResponse(text=text)

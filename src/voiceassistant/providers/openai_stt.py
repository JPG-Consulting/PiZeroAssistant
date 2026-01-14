"""OpenAI STT provider."""

from __future__ import annotations

from typing import Optional

import requests

from voiceassistant.providers.base import Provider, ProviderError
from voiceassistant.providers.http import STTResponse
from voiceassistant.providers.stt import STTProvider


class OpenAISTTProvider(Provider, STTProvider):
    """OpenAI STT provider using multipart/form-data."""

    # Non-streaming invariant: upload the full WAV and return the full transcript.
    # Streaming STT must be a separate provider type.

    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout_s: float,
        api_key: Optional[str],
        model: str,
    ) -> None:
        super().__init__(name, timeout_s, api_key)
        self.endpoint = endpoint
        self.model = model

    def _headers(self) -> dict:
        headers = {"User-Agent": "VoiceAssistant/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def transcribe(self, wav_bytes: bytes) -> STTResponse:
        # OpenAI STT is configured for WAV only; config validation enforces this.
        # Validate early to provide clearer failure modes.
        if not isinstance(wav_bytes, (bytes, bytearray)):
            raise ProviderError("STT input must be WAV bytes")
        if len(wav_bytes) < 12 or wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            raise ProviderError("STT input must be WAV bytes")

        files = {
            "file": ("audio.wav", bytes(wav_bytes), "audio/wav"),
        }
        data = {"model": self.model}
        try:
            resp = requests.post(
                self.endpoint,
                headers=self._headers(),
                files=files,
                data=data,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code != 200:
            snippet = " ".join((resp.text or "").split())
            if len(snippet) > 200:
                snippet = f"{snippet[:200]}..."
            message = f"HTTP {resp.status_code}"
            if snippet:
                message = f"{message}: {snippet}"
            raise ProviderError(message)
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            raise ProviderError(f"Unexpected response content type: {content_type or 'unknown'}")
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
        return STTResponse(text=text)

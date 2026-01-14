"""HTTP-based providers for STT, LLM, and TTS."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import requests

from voiceassistant.audio.stream import AudioStream, BytesAudioStream
from voiceassistant.logging_config import get_logger
from voiceassistant.providers.base import LLMRequest, Provider, ProviderError, build_messages_with_system
from voiceassistant.providers.stt import STTProvider
from voiceassistant.providers.tts import TTSProvider

logger = get_logger(__name__)


@dataclass
class STTResponse:
    text: str


@dataclass
class LLMResponse:
    text: str


@dataclass
class TTSResponse:
    audio: Optional[AudioStream] = None
    wav_bytes: Optional[bytes] = None

    @classmethod
    def from_bytes(cls, wav_bytes: bytes, sample_rate_hz: int, channels: int) -> "TTSResponse":
        audio = BytesAudioStream(
            data=wav_bytes,
            format="wav",
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )
        return cls(audio=audio, wav_bytes=wav_bytes)


class HttpProvider(Provider):
    def __init__(self, name: str, endpoint: str, timeout_s: float, api_key: Optional[str]) -> None:
        super().__init__(name, timeout_s, api_key)
        self.endpoint = endpoint

    def _headers(self) -> dict:
        headers = {"User-Agent": "VoiceAssistant/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class HttpSTTProvider(HttpProvider, STTProvider):
    def transcribe(self, wav_bytes: bytes) -> STTResponse:
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
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
            raise ProviderError(f"HTTP {resp.status_code}")
        data = resp.json()
        text = data.get("text")
        if not text:
            raise ProviderError("Missing text in STT response")
        return STTResponse(text=text)


class HttpLLMProvider(HttpProvider):
    def complete(self, req: LLMRequest) -> LLMResponse:
        payload = {"messages": build_messages_with_system(req.messages, req.system_prompt)}
        endpoint = self.endpoint
        try:
            resp = requests.post(
                endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise ProviderError(str(exc)) from exc
        if resp.status_code != 200:
            raise ProviderError(f"HTTP {resp.status_code}")
        data = resp.json()
        text = data.get("text")
        if not text:
            raise ProviderError("Missing text in LLM response")
        return LLMResponse(text=text)


class HttpTTSProvider(HttpProvider, TTSProvider):
    def synthesize(self, text: str) -> TTSResponse:
        payload = {"text": text}
        try:
            resp = requests.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise ProviderError(str(exc)) from exc
        if resp.status_code != 200:
            raise ProviderError(f"HTTP {resp.status_code}")
        if resp.headers.get("Content-Type", "").startswith("audio/"):
            return TTSResponse(wav_bytes=resp.content)
        data = resp.json()
        audio_b64 = data.get("audio_wav_base64")
        if not audio_b64:
            raise ProviderError("Missing audio_wav_base64 in TTS response")
        try:
            wav_bytes = base64.b64decode(audio_b64)
        except (ValueError, TypeError) as exc:
            raise ProviderError("Invalid base64 audio") from exc
        return TTSResponse(wav_bytes=wav_bytes)

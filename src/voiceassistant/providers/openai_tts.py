"""OpenAI Audio Speech TTS provider."""

from __future__ import annotations

from typing import Iterator, Optional

import requests

from voiceassistant.audio.stream import AudioStream
from voiceassistant.providers.base import Provider, ProviderError
from voiceassistant.providers.http import TTSResponse
from voiceassistant.providers.tts import TTSProvider


class OpenAITTSProvider(Provider, TTSProvider):
    """OpenAI TTS provider using the Audio → Speech API."""

    _DEFAULT_SAMPLE_RATE_HZ = 24000
    _DEFAULT_CHANNELS = 1

    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout_s: float,
        api_key: Optional[str],
        model: str,
        voice: Optional[str],
    ) -> None:
        super().__init__(name, timeout_s, api_key)
        self.endpoint = endpoint
        self.model = model
        self.voice = voice  # Intentionally no default to avoid silent behavior changes.

    def _headers(self) -> dict:
        headers = {
            "User-Agent": "VoiceAssistant/1.0",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def synthesize(self, text: str) -> TTSResponse:
        if not isinstance(text, str):
            raise ProviderError("TTS input must be a string")
        if not text.strip():
            raise ProviderError("TTS input must be non-empty")

        payload = {
            "model": self.model,
            "input": text,
        }
        if self.voice:
            payload["voice"] = self.voice
        try:
            resp = requests.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_s,
                stream=True,
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
            resp.close()
            raise ProviderError(message)

        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        audio_format = self._infer_format(content_type)
        audio = self._build_streaming_audio(
            response=resp,
            audio_format=audio_format,
            sample_rate_hz=self._DEFAULT_SAMPLE_RATE_HZ,
            channels=self._DEFAULT_CHANNELS,
        )
        return TTSResponse(audio=audio, wav_bytes=None)

    @staticmethod
    def _infer_format(content_type: str) -> str:
        if content_type in {"audio/wav", "audio/wave", "audio/x-wav"}:
            return "wav"
        if content_type in {"audio/mpeg", "audio/mp3"}:
            return "mp3"
        if content_type in {"audio/opus", "audio/ogg"}:
            return "opus"
        if content_type in {"audio/pcm", "audio/raw"}:
            return "pcm"
        if content_type == "application/octet-stream":
            return "pcm"
        return "pcm"

    def _build_streaming_audio(
        self,
        response: requests.Response,
        audio_format: str,
        sample_rate_hz: int,
        channels: int,
        chunk_size: int = 4096,
    ) -> AudioStream:
        iterator = response.iter_content(chunk_size=chunk_size)
        return _ResponseAudioStream(
            response=response,
            iterator=iterator,
            format=audio_format,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )


class _ResponseAudioStream(AudioStream):
    def __init__(
        self,
        response: requests.Response,
        iterator,
        format: str,
        sample_rate_hz: int,
        channels: int,
    ) -> None:
        self._response = response
        self._iterator = iterator
        self.format = format
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self._closed = False

    def iter_chunks(self) -> Iterator[bytes]:
        try:
            for chunk in self._iterator:
                if chunk:
                    yield chunk
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._response.close()

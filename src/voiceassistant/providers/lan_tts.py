"""LAN OpenAI-compatible TTS provider."""

from __future__ import annotations

import io
import wave
from typing import Iterator, Optional

import requests

from voiceassistant.audio.stream import AudioStream
from voiceassistant.logging_config import get_logger
from voiceassistant.providers.base import ProviderError
from voiceassistant.providers.http import HttpProvider, TTSResponse
from voiceassistant.providers.tts import TTSProvider


logger = get_logger(__name__)


class LanHttpTTSProvider(HttpProvider, TTSProvider):
    """LAN OpenAI-compatible TTS provider.

    Streaming is transport-level only; playback handles decoding and buffering.
    This provider exposes raw encoded audio streams without decoding.
    """
    _DEFAULT_SAMPLE_RATE_HZ = 16000
    _DEFAULT_CHANNELS = 1

    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout_s: float,
        api_key: Optional[str],
    ) -> None:
        super().__init__(name, endpoint, timeout_s, api_key)

    def synthesize(self, text: str) -> TTSResponse:
        if not isinstance(text, str):
            raise ProviderError("TTS input must be a string")
        if not text.strip():
            raise ProviderError("TTS input must be non-empty")

        payload = {"input": text, "format": "pcm"}
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
            snippet = resp.text.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = f"{snippet[:200]}..."
            message = f"HTTP {resp.status_code}"
            if snippet:
                message = f"{message}: {snippet}"
            resp.close()
            raise ProviderError(message)

        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        audio_format = self._infer_format(content_type)
        logger.debug(
            "LAN TTS content-type=%s inferred_format=%s",
            content_type,
            audio_format,
        )
        sample_rate_hz = self._parse_int_header(
            resp.headers.get("X-Audio-Sample-Rate"),
            self._DEFAULT_SAMPLE_RATE_HZ,
        )
        channels = self._parse_int_header(
            resp.headers.get("X-Audio-Channels"),
            self._DEFAULT_CHANNELS,
        )
        audio = self._build_streaming_audio(
            response=resp,
            audio_format=audio_format,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
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

    @staticmethod
    def _parse_int_header(value: Optional[str], fallback: int) -> int:
        if value is None:
            return fallback
        try:
            return int(value)
        except ValueError:
            return fallback

    def _build_streaming_audio(
        self,
        response: requests.Response,
        audio_format: str,
        sample_rate_hz: int,
        channels: int,
        chunk_size: int = 4096,
    ) -> AudioStream:
        iterator = response.iter_content(chunk_size=chunk_size)
        prefix = b""
        if audio_format == "wav":
            prefix, sample_rate_hz, channels = self._prime_wav_stream(
                iterator,
                sample_rate_hz,
                channels,
            )
        return _ResponseAudioStream(
            response=response,
            iterator=iterator,
            prefix=prefix,
            format=audio_format,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )

    @staticmethod
    def _prime_wav_stream(
        iterator: Iterator[bytes],
        default_rate: int,
        default_channels: int,
        max_bytes: int = 65536,
    ) -> tuple[bytes, int, int]:
        buffer = bytearray()
        parsed_rate = default_rate
        parsed_channels = default_channels
        while len(buffer) < max_bytes:
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            if chunk:
                buffer.extend(chunk)
            if len(buffer) < 44:
                continue
            try:
                with wave.open(io.BytesIO(buffer), "rb") as handle:
                    parsed_rate = handle.getframerate()
                    parsed_channels = handle.getnchannels()
                    return bytes(buffer), parsed_rate, parsed_channels
            except wave.Error:
                continue
        return bytes(buffer), parsed_rate, parsed_channels


class _ResponseAudioStream:
    def __init__(
        self,
        response: requests.Response,
        iterator,
        prefix: bytes,
        format: str,
        sample_rate_hz: int,
        channels: int,
    ) -> None:
        self._response = response
        self._iterator = iterator
        self._prefix = prefix
        self.format = format
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self._closed = False

    def iter_chunks(self) -> Iterator[bytes]:
        chunk_count = 0
        total_bytes = 0
        try:
            if self._prefix:
                chunk_count += 1
                total_bytes += len(self._prefix)
                logger.debug(
                    "LAN TTS chunk %d size=%d bytes",
                    chunk_count,
                    len(self._prefix),
                )
                yield self._prefix
                self._prefix = b""
            for chunk in self._iterator:
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    if chunk_count <= 10 or chunk_count % 50 == 0:
                        logger.debug(
                            "LAN TTS chunk %d size=%d bytes",
                            chunk_count,
                            len(chunk),
                        )
                    yield chunk
        finally:
            logger.debug(
                "LAN TTS stream completed chunks=%d total_bytes=%d",
                chunk_count,
                total_bytes,
            )
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        logger.debug("LAN TTS stream closing response")
        self._closed = True
        self._response.close()

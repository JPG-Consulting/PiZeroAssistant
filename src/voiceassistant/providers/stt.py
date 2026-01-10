"""STT provider interface."""

from __future__ import annotations

from typing import Protocol

from voiceassistant.providers.http import STTResponse


class STTProvider(Protocol):
    def transcribe(self, wav_bytes: bytes) -> STTResponse:
        ...

"""STT provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceassistant.providers.http import STTResponse


class STTProvider(ABC):
    """STT provider contract for typing and invariants only."""

    # Typing- and invariants-only interface: must never include shared logic, defaults, helpers, or behavior.

    @abstractmethod
    def transcribe(self, wav_bytes: bytes) -> "STTResponse":
        ...

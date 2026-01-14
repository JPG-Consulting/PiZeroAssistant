"""TTS provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceassistant.providers.http import TTSResponse


class TTSProvider(ABC):
    """TTS provider contract for typing and invariants only."""

    # Typing- and invariants-only interface: must never include shared logic, defaults, helpers, or behavior.

    @abstractmethod
    def synthesize(self, text: str) -> "TTSResponse":
        ...

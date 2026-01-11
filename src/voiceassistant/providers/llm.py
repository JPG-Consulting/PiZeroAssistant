"""LLM provider interface."""

from __future__ import annotations

from typing import Protocol

from voiceassistant.providers.base import LLMRequest
from voiceassistant.providers.http import LLMResponse


class LLMProvider(Protocol):
    def complete(self, req: LLMRequest) -> LLMResponse:
        ...

"""Diagnostic local-echo LLM provider."""

from __future__ import annotations

from voiceassistant.providers.base import LLMRequest, Provider, ProviderError, build_messages_with_system
from voiceassistant.providers.http import LLMResponse


class LocalEchoLLMProvider(Provider):
    """Diagnostic-only LLM provider that echoes the last user message."""

    def __init__(self, name: str) -> None:
        self.name = name

    def complete(self, req: LLMRequest) -> LLMResponse:
        messages = build_messages_with_system(req.messages, req.system_prompt)
        if not isinstance(messages, list):
            raise ProviderError("LLM request messages must be a list")
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return LLMResponse(text=message["content"])
        return LLMResponse(text="I didn't receive any input.")

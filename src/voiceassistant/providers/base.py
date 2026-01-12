"""Provider base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMRequest:
    messages: list[dict]
    system_prompt: str


def build_messages_with_system(messages: list[dict], system_prompt: str) -> list[dict]:
    if not system_prompt:
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


@dataclass
class ProviderHealth:
    failures: int = 0
    cooldown_until: float = 0.0


class ProviderError(RuntimeError):
    """Raised when a provider fails."""


class Provider:
    """Base provider interface."""

    def __init__(self, name: str, timeout_s: float, api_key: Optional[str]) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.api_key = api_key

    def health_check(self) -> bool:
        return True

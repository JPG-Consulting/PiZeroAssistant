"""Provider routing with fallback and health checks."""

# This module must depend only on provider interfaces (STTProvider / LLMProvider / TTSProvider).
# Concrete provider implementations must not be imported here.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar

from voiceassistant.logging_config import get_logger
from voiceassistant.providers.base import Provider, ProviderError, ProviderHealth

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RoutedProvider:
    provider: Provider
    max_failures: int
    cooldown_s: int


class ProviderRouter:
    def __init__(self, providers: List[RoutedProvider], max_fallbacks: int) -> None:
        self._providers = providers
        self._max_fallbacks = max_fallbacks
        self._health: Dict[str, ProviderHealth] = {
            p.provider.name: ProviderHealth() for p in providers
        }

    def _eligible(self) -> Iterable[RoutedProvider]:
        now = time.monotonic()
        for provider in self._providers:
            health = self._health[provider.provider.name]
            if now < health.cooldown_until:
                continue
            if not provider.provider.health_check():
                continue
            yield provider

    def call(self, fn: Callable[[Provider], T]) -> Tuple[T, str]:
        errors = []
        for attempt, routed in enumerate(self._eligible()):
            if attempt > self._max_fallbacks:
                break
            try:
                result = fn(routed.provider)
                self._health[routed.provider.name].failures = 0
                return result, routed.provider.name
            except ProviderError as exc:
                health = self._health[routed.provider.name]
                health.failures += 1
                if health.failures >= routed.max_failures:
                    health.cooldown_until = time.monotonic() + routed.cooldown_s
                errors.append(f"{routed.provider.name}: {exc}")
        raise ProviderError("All providers failed: " + "; ".join(errors))

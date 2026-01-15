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

    def has_streaming_provider(self) -> bool:
        # Evaluates streaming support against currently eligible providers (health/cooldown applied).
        return any(
            callable(getattr(routed.provider, "stream", None))
            for routed in self._eligible()
        )

    def _record_success(self, provider_name: str, count_failures: bool) -> None:
        if count_failures:
            self._health[provider_name].failures = 0

    def _record_failure(
        self,
        provider_name: str,
        *,
        count_failures: bool,
        max_failures: int,
        cooldown_s: int,
    ) -> None:
        if not count_failures:
            return
        health = self._health[provider_name]
        health.failures += 1
        if health.failures >= max_failures:
            health.cooldown_until = time.monotonic() + cooldown_s

    def call(self, fn: Callable[[Provider], T], *, count_failures: bool = True) -> Tuple[T, str]:
        errors = []
        for attempt, routed in enumerate(self._eligible()):
            if attempt > self._max_fallbacks:
                break
            try:
                result = fn(routed.provider)
                self._record_success(routed.provider.name, count_failures)
                return result, routed.provider.name
            except ProviderError as exc:
                self._record_failure(
                    routed.provider.name,
                    count_failures=count_failures,
                    max_failures=routed.max_failures,
                    cooldown_s=routed.cooldown_s,
                )
                errors.append(f"{routed.provider.name}: {exc}")
        raise ProviderError("All providers failed: " + "; ".join(errors))

    def call_streaming(self, fn: Callable[[Provider], T]) -> Tuple[T, str]:
        errors = []
        attempts = 0
        for routed in self._eligible():
            if not callable(getattr(routed.provider, "stream", None)):
                continue
            if attempts > self._max_fallbacks:
                break
            try:
                result = fn(routed.provider)
                self._record_success(routed.provider.name, count_failures=True)
                return result, routed.provider.name
            except ProviderError as exc:
                self._record_failure(
                    routed.provider.name,
                    count_failures=True,
                    max_failures=routed.max_failures,
                    cooldown_s=routed.cooldown_s,
                )
                errors.append(f"{routed.provider.name}: {exc}")
            attempts += 1
        if errors:
            raise ProviderError("All providers failed: " + "; ".join(errors))
        raise ProviderError("No streaming providers available")

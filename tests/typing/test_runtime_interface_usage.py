from __future__ import annotations

from typing import assert_type, cast

from voiceassistant.providers.base import LLMRequest, Provider
from voiceassistant.providers.http import LLMResponse, STTResponse, TTSResponse
from voiceassistant.providers.llm import LLMProvider
from voiceassistant.providers.router import ProviderRouter, RoutedProvider
from voiceassistant.providers.stt import STTProvider
from voiceassistant.providers.tts import TTSProvider


class DummySTTProvider(Provider, STTProvider):
    def transcribe(self, wav_bytes: bytes) -> STTResponse:
        raise NotImplementedError


class DummyLLMProvider(Provider, LLMProvider):
    def complete(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class DummyTTSProvider(Provider, TTSProvider):
    def synthesize(self, text: str) -> TTSResponse:
        raise NotImplementedError


def _router_for(provider: Provider) -> ProviderRouter:
    routed = RoutedProvider(provider=provider, max_failures=1, cooldown_s=1)
    return ProviderRouter([routed], max_fallbacks=0)


def test_router_accepts_stt_interface() -> None:
    provider = DummySTTProvider(name="stt", timeout_s=1.0, api_key=None)
    router = _router_for(provider)
    result, _ = router.call(lambda current: cast(STTProvider, current))
    assert_type(result, STTProvider)


def test_router_accepts_llm_interface() -> None:
    provider = DummyLLMProvider(name="llm", timeout_s=1.0, api_key=None)
    router = _router_for(provider)
    result, _ = router.call(lambda current: cast(LLMProvider, current))
    assert_type(result, LLMProvider)


def test_router_accepts_tts_interface() -> None:
    provider = DummyTTSProvider(name="tts", timeout_s=1.0, api_key=None)
    router = _router_for(provider)
    result, _ = router.call(lambda current: cast(TTSProvider, current))
    assert_type(result, TTSProvider)

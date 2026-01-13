from __future__ import annotations

from typing import assert_type

from voiceassistant.config import ProviderConfig
from voiceassistant.providers.factory import build_llm_provider, build_stt_provider, build_tts_provider
from voiceassistant.providers.llm import LLMProvider
from voiceassistant.providers.stt import STTProvider
from voiceassistant.providers.tts import TTSProvider


def _base_config(
    *,
    name: str,
    provider_type: str,
    endpoint: str,
    audio_formats: list[str] | None = None,
    model: str | None = None,
    voice: str | None = None,
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        provider_type=provider_type,
        endpoint=endpoint,
        timeout_s=5.0,
        api_key_env=None,
        max_failures=2,
        cooldown_s=60,
        audio_formats=audio_formats,
        max_tokens_per_request=None,
        model=model,
        voice=voice,
    )


def test_factory_return_types() -> None:
    stt_cfg = _base_config(
        name="stt",
        provider_type="http",
        endpoint="http://localhost/stt",
        audio_formats=["wav"],
    )
    stt = build_stt_provider(stt_cfg)
    assert_type(stt, STTProvider)

    llm_cfg = _base_config(
        name="llm",
        provider_type="local_echo",
        endpoint="",
    )
    llm = build_llm_provider(llm_cfg)
    assert_type(llm, LLMProvider)

    tts_cfg = _base_config(
        name="tts",
        provider_type="http",
        endpoint="http://localhost/tts",
    )
    tts = build_tts_provider(tts_cfg)
    assert_type(tts, TTSProvider)

"""Provider factory for STT, LLM, and TTS providers."""

from __future__ import annotations

from voiceassistant.config import ConfigError, ProviderConfig, resolve_api_key
from voiceassistant.providers.http import HttpLLMProvider, HttpSTTProvider, HttpTTSProvider
from voiceassistant.providers.lan_stt import LanHttpSTTProvider
from voiceassistant.providers.stt import STTProvider


def build_stt_provider(config: ProviderConfig) -> STTProvider:
    if config.provider_type == "http":
        return HttpSTTProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
        )
    if config.provider_type == "lan_http":
        return LanHttpSTTProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
        )
    # Should be unreachable because config.py validates provider_type; defensive only.
    raise ConfigError(
        f"Unsupported STT provider_type '{config.provider_type}' for '{config.name}'"
    )


def build_llm_provider(config: ProviderConfig) -> HttpLLMProvider:
    if config.provider_type == "http":
        return HttpLLMProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
        )
    # Should be unreachable because config.py validates provider_type; defensive only.
    raise ConfigError(
        f"Unsupported LLM provider_type '{config.provider_type}' for '{config.name}'"
    )


def build_tts_provider(config: ProviderConfig) -> HttpTTSProvider:
    if config.provider_type == "http":
        return HttpTTSProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
        )
    # Should be unreachable because config.py validates provider_type; defensive only.
    raise ConfigError(
        f"Unsupported TTS provider_type '{config.provider_type}' for '{config.name}'"
    )

"""Provider factory for STT, LLM, and TTS providers."""

from __future__ import annotations

from voiceassistant.config import ConfigError, ProviderConfig, resolve_api_key
from voiceassistant.providers.echo_llm import LocalEchoLLMProvider
from voiceassistant.providers.http import HttpLLMProvider, HttpSTTProvider, HttpTTSProvider
from voiceassistant.providers.lan_llm import LanHttpLLMProvider
from voiceassistant.providers.lan_stt import LanHttpSTTProvider
from voiceassistant.providers.lan_tts import LanHttpTTSProvider
from voiceassistant.providers.llm import LLMProvider
from voiceassistant.providers.openai_llm import OpenAILLMProvider
from voiceassistant.providers.openai_tts import OpenAITTSProvider
from voiceassistant.providers.stt import STTProvider
from voiceassistant.providers.tts import TTSProvider


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


def build_llm_provider(config: ProviderConfig) -> LLMProvider:
    if config.provider_type == "http":
        return HttpLLMProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
        )
    if config.provider_type == "local_echo":
        return LocalEchoLLMProvider(name=config.name)
    if config.provider_type == "lan_http":
        return LanHttpLLMProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
        )
    if config.provider_type == "openai":
        return OpenAILLMProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
            model=config.model or "",
        )
    # Should be unreachable because config.py validates provider_type; defensive only.
    raise ConfigError(
        f"Unsupported LLM provider_type '{config.provider_type}' for '{config.name}'"
    )


def build_tts_provider(config: ProviderConfig) -> TTSProvider:
    """Return a TTSProvider contract with synthesize(text) -> TTSResponse only."""
    if config.provider_type == "http":
        return HttpTTSProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
        )
    if config.provider_type == "lan_http":
        return LanHttpTTSProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
        )
    if config.provider_type == "openai":
        return OpenAITTSProvider(
            name=config.name,
            endpoint=config.endpoint,
            timeout_s=config.timeout_s,
            api_key=resolve_api_key(config),
            model=config.model or "",
            voice=config.voice,
        )
    # Should be unreachable because config.py validates provider_type; defensive only.
    raise ConfigError(
        f"Unsupported TTS provider_type '{config.provider_type}' for '{config.name}'"
    )

"""Configuration loading for Voice Assistant."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int
    channels: int
    frame_duration_ms: int
    device: Optional[str]
    pre_roll_ms: int
    max_record_seconds: int
    record_silence_ms: int
    vad_mode: int


@dataclass(frozen=True)
class WakewordConfig:
    model_path: str
    inference_window_ms: int
    score_threshold: float
    wakeword_cooldown_ms: int


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    provider_type: str
    endpoint: str
    timeout_s: float
    api_key_env: Optional[str]
    max_failures: int
    cooldown_s: int
    audio_formats: Optional[List[str]]


@dataclass(frozen=True)
class RoutingConfig:
    stt_providers: List[ProviderConfig]
    llm_providers: List[ProviderConfig]
    tts_providers: List[ProviderConfig]
    max_fallbacks: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    module_levels: Dict[str, str]


@dataclass(frozen=True)
class ConversationPersistenceConfig:
    enabled: bool
    path: Optional[str]


@dataclass(frozen=True)
class ConversationConfig:
    enabled: bool
    max_turns: int
    max_chars: int
    reset_after_idle_s: int
    reset_commands: List[str]
    persistence: ConversationPersistenceConfig


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig
    wakeword: WakewordConfig
    routing: RoutingConfig
    logging: LoggingConfig
    conversation: ConversationConfig
    wake_beep_path: Optional[str]
    record_output_path: str


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


SUPPORTED_PROVIDER_TYPES = {
    "stt": {"http", "lan_http"},
    "llm": {"http"},
    "tts": {"http"},
}

STT_AUDIO_FORMATS = {"wav", "opus", "mp3"}


def _require(value: Any, field: str) -> Any:
    if value is None:
        raise ConfigError(f"Missing required config field: {field}")
    return value


def _normalize_audio_formats(raw_formats: Any, field_path: str) -> List[str]:
    if raw_formats is None:
        return ["wav"]
    if not isinstance(raw_formats, list):
        raise ConfigError(f"{field_path} must be a list of audio format strings")
    formats = [str(item).lower() for item in raw_formats]
    unknown = sorted({fmt for fmt in formats if fmt not in STT_AUDIO_FORMATS})
    if unknown:
        allowed = ", ".join(sorted(STT_AUDIO_FORMATS))
        raise ConfigError(
            f"{field_path} contains unsupported formats: {', '.join(unknown)}. "
            f"Supported formats: {allowed}"
        )
    if "wav" not in formats:
        raise ConfigError(f"{field_path} must include 'wav' as a required baseline")
    return formats


def _provider_list(raw_list: List[Dict[str, Any]], service: str) -> List[ProviderConfig]:
    providers: List[ProviderConfig] = []
    for index, item in enumerate(raw_list):
        base_path = f"routing.{service}_providers[{index}]"
        name = _require(item.get("name"), f"{base_path}.name")
        provider_type = _require(item.get("provider_type"), f"{base_path}.provider_type")
        provider_type = str(provider_type).lower()
        supported = SUPPORTED_PROVIDER_TYPES.get(service, set())
        if provider_type not in supported:
            supported_types = ", ".join(sorted(supported)) or "<none>"
            raise ConfigError(
                f"Unsupported {service} provider_type '{provider_type}' for '{name}'. "
                f"Supported types: {supported_types}"
            )
        if service == "stt":
            audio_formats = _normalize_audio_formats(
                item.get("audio_formats"),
                f"{base_path}.audio_formats",
            )
        else:
            if item.get("audio_formats") is not None:
                raise ConfigError(
                    f"{base_path}.audio_formats is only valid for stt providers"
                )
            audio_formats = None
        providers.append(
            ProviderConfig(
                name=name,
                provider_type=provider_type,
                endpoint=_require(item.get("endpoint"), f"{base_path}.endpoint"),
                timeout_s=float(item.get("timeout_s", 15.0)),
                api_key_env=item.get("api_key_env"),
                max_failures=int(item.get("max_failures", 2)),
                cooldown_s=int(item.get("cooldown_s", 60)),
                audio_formats=audio_formats,
            )
        )
    return providers


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    audio = raw.get("audio", {})
    wakeword = raw.get("wakeword", {})
    routing = raw.get("routing", {})
    logging_cfg = raw.get("logging", {})
    conversation = raw.get("conversation", {})
    persistence = conversation.get("persistence", {})
    persistence_enabled = bool(persistence.get("enabled", False))
    persistence_path = persistence.get("path")
    if persistence_enabled and not persistence_path:
        raise ConfigError("conversation.persistence.path is required when persistence is enabled")

    raw_reset_commands = conversation.get(
        "reset_commands",
        ["reset conversation", "forget this", "clear memory"],
    )
    reset_commands = [str(command) for command in raw_reset_commands]

    config = AppConfig(
        audio=AudioConfig(
            sample_rate_hz=int(_require(audio.get("sample_rate_hz"), "audio.sample_rate_hz")),
            channels=int(audio.get("channels", 1)),
            frame_duration_ms=int(audio.get("frame_duration_ms", 20)),
            device=audio.get("device"),
            pre_roll_ms=int(audio.get("pre_roll_ms", 400)),
            max_record_seconds=int(audio.get("max_record_seconds", 8)),
            record_silence_ms=int(audio.get("record_silence_ms", 800)),
            vad_mode=int(audio.get("vad_mode", 2)),
        ),
        wakeword=WakewordConfig(
            model_path=str(_require(wakeword.get("model_path"), "wakeword.model_path")),
            inference_window_ms=int(wakeword.get("inference_window_ms", 80)),
            score_threshold=float(wakeword.get("score_threshold", 0.6)),
            wakeword_cooldown_ms=int(wakeword.get("wakeword_cooldown_ms", 1500)),
        ),
        routing=RoutingConfig(
            stt_providers=_provider_list(routing.get("stt_providers", []), "stt"),
            llm_providers=_provider_list(routing.get("llm_providers", []), "llm"),
            tts_providers=_provider_list(routing.get("tts_providers", []), "tts"),
            max_fallbacks=int(routing.get("max_fallbacks", 2)),
        ),
        logging=LoggingConfig(
            level=str(logging_cfg.get("level", "INFO")),
            module_levels=dict(logging_cfg.get("module_levels", {})),
        ),
        conversation=ConversationConfig(
            enabled=bool(conversation.get("enabled", True)),
            max_turns=int(conversation.get("max_turns", 6)),
            max_chars=int(conversation.get("max_chars", 4000)),
            reset_after_idle_s=int(conversation.get("reset_after_idle_s", 300)),
            reset_commands=reset_commands,
            persistence=ConversationPersistenceConfig(
                enabled=persistence_enabled,
                path=str(persistence_path) if persistence_path is not None else None,
            ),
        ),
        wake_beep_path=raw.get("wake_beep_path"),
        record_output_path=str(raw.get("record_output_path", "/tmp/last_command.wav")),
    )

    if config.audio.frame_duration_ms not in (10, 20, 30):
        raise ConfigError("audio.frame_duration_ms must be 10, 20, or 30 ms for VAD")
    if config.audio.sample_rate_hz != 16000:
        raise ConfigError("audio.sample_rate_hz must be 16000 Hz for openWakeWord")
    if config.wakeword.inference_window_ms < 25:
        raise ConfigError("wakeword.inference_window_ms must be >= 25 ms")
    if config.conversation.max_turns < 0:
        raise ConfigError("conversation.max_turns must be >= 0")
    if config.conversation.max_chars < 0:
        raise ConfigError("conversation.max_chars must be >= 0")
    if config.conversation.reset_after_idle_s < 0:
        raise ConfigError("conversation.reset_after_idle_s must be >= 0")

    return config


def resolve_api_key(config: ProviderConfig) -> Optional[str]:
    if not config.api_key_env:
        return None
    return os.environ.get(config.api_key_env)

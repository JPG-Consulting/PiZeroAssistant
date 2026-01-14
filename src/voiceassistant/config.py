"""Configuration loading for Voice Assistant."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
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
    tts_prewarm: "TTSPrewarmConfig"


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
    max_tokens_per_request: Optional[int]
    model: Optional[str]
    voice: Optional[str]


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


@dataclass(frozen=True)
class TTSPrewarmConfig:
    enabled: bool


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


SUPPORTED_PROVIDER_TYPES = {
    "stt": {"http", "lan_http", "openai"},
    "llm": {"http", "lan_http", "local_echo", "openai"},
    "tts": {"http", "lan_http", "openai"},
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
        max_tokens_value = item.get("max_tokens_per_request")
        if service != "llm" and max_tokens_value is not None:
            raise ConfigError(
                f"{base_path}.max_tokens_per_request is only valid for llm providers"
            )
        if max_tokens_value is None:
            max_tokens_per_request = None
        else:
            max_tokens_per_request = int(max_tokens_value)
            if max_tokens_per_request <= 0:
                raise ConfigError(
                    f"{base_path}.max_tokens_per_request must be a positive integer"
                )
        model_value = item.get("model")
        voice_value = item.get("voice")
        if service == "llm":
            model = str(model_value) if model_value is not None else None
            if voice_value is not None:
                raise ConfigError(f"{base_path}.voice is only valid for tts providers")
            if provider_type == "openai":
                if not model or not model.strip():
                    raise ConfigError(f"{base_path}.model is required for openai providers")
                api_key_env = item.get("api_key_env")
                if not api_key_env or not str(api_key_env).strip():
                    raise ConfigError(f"{base_path}.api_key_env is required for openai providers")
            voice = None
        elif service == "tts":
            if model_value is not None:
                model = str(model_value)
            else:
                model = None
            if voice_value is None:
                voice = None
            else:
                voice = str(voice_value)
                if not voice.strip():
                    raise ConfigError(f"{base_path}.voice must be a non-empty string")
            if provider_type == "openai":
                if not model or not model.strip():
                    raise ConfigError(f"{base_path}.model is required for openai providers")
                api_key_env = item.get("api_key_env")
                if not api_key_env or not str(api_key_env).strip():
                    raise ConfigError(f"{base_path}.api_key_env is required for openai providers")
            elif model_value is not None:
                raise ConfigError(f"{base_path}.model is only valid for openai tts providers")
        else:
            if provider_type == "openai":
                model = str(model_value) if model_value is not None else None
                if not model or not model.strip():
                    raise ConfigError(f"{base_path}.model is required for openai providers")
                api_key_env = item.get("api_key_env")
                if not api_key_env or not str(api_key_env).strip():
                    raise ConfigError(f"{base_path}.api_key_env is required for openai providers")
                if audio_formats != ["wav"]:
                    raise ConfigError(f"{base_path}.audio_formats must be ['wav'] for openai stt providers")
            else:
                if model_value is not None:
                    raise ConfigError(
                        f"{base_path}.model is only valid for llm or openai tts providers"
                    )
                if voice_value is not None:
                    raise ConfigError(f"{base_path}.voice is only valid for tts providers")
                model = None
            voice = None
        endpoint_value = item.get("endpoint")
        if service == "llm" and provider_type == "local_echo":
            if endpoint_value is not None and str(endpoint_value).strip() != "":
                raise ConfigError(
                    f"{base_path}.endpoint must be omitted or empty for local_echo providers"
                )
            if item.get("api_key_env") is not None:
                raise ConfigError(
                    f"{base_path}.api_key_env must be null for local_echo providers"
                )
            endpoint = ""
        else:
            endpoint = _require(endpoint_value, f"{base_path}.endpoint")
        providers.append(
            ProviderConfig(
                name=name,
                provider_type=provider_type,
                endpoint=endpoint,
                timeout_s=float(item.get("timeout_s", 15.0)),
                api_key_env=item.get("api_key_env"),
                max_failures=int(item.get("max_failures", 2)),
                cooldown_s=int(item.get("cooldown_s", 60)),
                audio_formats=audio_formats,
                max_tokens_per_request=max_tokens_per_request,
                model=model,
                voice=voice,
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
    wakeword_model_path = _require(wakeword.get("model_path"), "wakeword.model_path")
    wake_beep_path = raw.get("wake_beep_path")
    record_output_path = raw.get("record_output_path", "/tmp/last_command.wav")

    if not Path(str(wakeword_model_path)).is_absolute():
        raise ConfigError(
            "wakeword.model_path must be an absolute path. Relative paths are not supported because "
            "services must not depend on the current working directory; provide an absolute path."
        )
    if wake_beep_path is not None and not Path(wake_beep_path).is_absolute():
        raise ConfigError(
            "wake_beep_path must be an absolute path. Relative paths are not supported because "
            "services must not depend on the current working directory; provide an absolute path."
        )
    if not Path(str(record_output_path)).is_absolute():
        raise ConfigError(
            "record_output_path must be an absolute path. Relative paths are not supported because "
            "services must not depend on the current working directory; provide an absolute path."
        )
    if persistence_enabled and not Path(str(persistence_path)).is_absolute():
        raise ConfigError(
            "conversation.persistence.path must be an absolute path. Relative paths are not supported because "
            "services must not depend on the current working directory; provide an absolute path."
        )

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
            tts_prewarm=TTSPrewarmConfig(
                enabled=bool(audio.get("tts_prewarm", {}).get("enabled", True))
            ),
        ),
        wakeword=WakewordConfig(
            model_path=str(wakeword_model_path),
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
        wake_beep_path=wake_beep_path,
        record_output_path=str(record_output_path),
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

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
    endpoint: str
    timeout_s: float
    api_key_env: Optional[str]
    max_failures: int
    cooldown_s: int


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
class AppConfig:
    audio: AudioConfig
    wakeword: WakewordConfig
    routing: RoutingConfig
    logging: LoggingConfig
    wake_beep_path: Optional[str]
    record_output_path: str


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


def _require(value: Any, field: str) -> Any:
    if value is None:
        raise ConfigError(f"Missing required config field: {field}")
    return value


def _provider_list(raw_list: List[Dict[str, Any]]) -> List[ProviderConfig]:
    providers: List[ProviderConfig] = []
    for item in raw_list:
        providers.append(
            ProviderConfig(
                name=_require(item.get("name"), "provider.name"),
                endpoint=_require(item.get("endpoint"), "provider.endpoint"),
                timeout_s=float(item.get("timeout_s", 15.0)),
                api_key_env=item.get("api_key_env"),
                max_failures=int(item.get("max_failures", 2)),
                cooldown_s=int(item.get("cooldown_s", 60)),
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
            stt_providers=_provider_list(routing.get("stt_providers", [])),
            llm_providers=_provider_list(routing.get("llm_providers", [])),
            tts_providers=_provider_list(routing.get("tts_providers", [])),
            max_fallbacks=int(routing.get("max_fallbacks", 2)),
        ),
        logging=LoggingConfig(
            level=str(logging_cfg.get("level", "INFO")),
            module_levels=dict(logging_cfg.get("module_levels", {})),
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

    return config


def resolve_api_key(config: ProviderConfig) -> Optional[str]:
    if not config.api_key_env:
        return None
    return os.environ.get(config.api_key_env)

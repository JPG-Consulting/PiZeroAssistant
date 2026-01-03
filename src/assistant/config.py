from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, Tuple
import yaml


@dataclass(frozen=True)
class AudioConfig:
    device: Optional[Any]
    sample_rate: int
    channels: int
    dtype: str
    block_ms: int


@dataclass(frozen=True)
class WebrtcVADConfig:
    aggressiveness: int


@dataclass(frozen=True)
class EnergyVADConfig:
    rms_threshold: float
    hangover_ms: int


@dataclass(frozen=True)
class VADConfig:
    type: str  # "webrtc" or "energy"
    webrtc: WebrtcVADConfig
    energy: EnergyVADConfig


@dataclass(frozen=True)
class FeatureConfig:
    n_mels: int
    win_ms: int
    hop_ms: int
    clip_seconds: float
    n_fft: int
    fmin: float
    fmax: float
    log_eps: float


@dataclass(frozen=True)
class SimulatedWakewordConfig:
    cooldown_sec: float
    trigger_probability: float


@dataclass(frozen=True)
class OpenWakeWordConfig:
    models: Tuple[str, ...]


@dataclass(frozen=True)
class WakewordConfig:
    type: str

    # ONNX-related (may be unused in simulated mode)
    model_path: str
    input_name: str
    output_name: str
    threshold: float
    consecutive_hits: int
    cooldown_ms: int

    # Simulated wake-word
    simulated: SimulatedWakewordConfig

    # OpenWakeWord
    openwakeword: OpenWakeWordConfig


@dataclass(frozen=True)
class RuntimeConfig:
    log_level: str


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig
    vad: VADConfig
    features: FeatureConfig
    wakeword: WakewordConfig
    runtime: RuntimeConfig


def _get(d: Dict[str, Any], key: str, default=None):
    return d[key] if key in d else default


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    audio = raw["audio"]
    vad = raw["vad"]
    feats = raw["features"]
    ww = raw["wakeword"]
    rt = raw.get("runtime", {})

    oww = ww.get("openwakeword", {})

    return AppConfig(
        audio=AudioConfig(
            device=_get(audio, "device", None),
            sample_rate=int(audio["sample_rate"]),
            channels=int(audio.get("channels", 1)),
            dtype=str(audio.get("dtype", "int16")),
            block_ms=int(audio.get("block_ms", 20)),
        ),
        vad=VADConfig(
            type=str(vad.get("type", "webrtc")),
            webrtc=WebrtcVADConfig(
                aggressiveness=int(vad.get("webrtc", {}).get("aggressiveness", 2))
            ),
            energy=EnergyVADConfig(
                rms_threshold=float(vad.get("energy", {}).get("rms_threshold", 0.015)),
                hangover_ms=int(vad.get("energy", {}).get("hangover_ms", 200)),
            ),
        ),
        features=FeatureConfig(
            n_mels=int(feats["n_mels"]),
            win_ms=int(feats["win_ms"]),
            hop_ms=int(feats["hop_ms"]),
            clip_seconds=float(feats["clip_seconds"]),
            n_fft=int(feats.get("n_fft", 1024)),
            fmin=float(feats.get("fmin", 20.0)),
            fmax=float(feats.get("fmax", 7600.0)),
            log_eps=float(feats.get("log_eps", 1.0e-6)),
        ),
        wakeword=WakewordConfig(
            type=str(ww.get("type", "simulated")),

            model_path=str(ww.get("model_path", "")),
            input_name=str(ww.get("input_name", "logmel")),
            output_name=str(ww.get("output_name", "logits")),
            threshold=float(ww.get("threshold", 0.80)),
            consecutive_hits=int(ww.get("consecutive_hits", 2)),
            cooldown_ms=int(ww.get("cooldown_ms", 1500)),

            simulated=SimulatedWakewordConfig(
                cooldown_sec=float(ww.get("simulated", {}).get("cooldown_sec", 5.0)),
                trigger_probability=float(ww.get("simulated", {}).get("trigger_probability", 0.02)),
            ),

            openwakeword=OpenWakeWordConfig(
                models=tuple(oww.get("models", ()))
            ),
        ),
        runtime=RuntimeConfig(
            log_level=str(rt.get("log_level", "INFO"))
        ),
    )

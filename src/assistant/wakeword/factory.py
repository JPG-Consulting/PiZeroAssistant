from assistant.wakeword.metrics import WakeWordMetrics
from assistant.wakeword.detector import SimulatedWakeWordDetector
from assistant.wakeword.onnx_detector import OnnxWakeWordDetector
from assistant.wakeword.openwakeword_detector import OpenWakeWordDetector


def create_wake_detector(cfg, metrics: WakeWordMetrics, mic):
    wake_cfg = cfg.wakeword

    if wake_cfg.type == "simulated":
        sim = wake_cfg.simulated
        return SimulatedWakeWordDetector(
            metrics=metrics,
            cooldown_sec=sim.cooldown_sec,
            trigger_probability=sim.trigger_probability,
        )

    if wake_cfg.type == "onnx":
        return OnnxWakeWordDetector(
            model_path=wake_cfg.model_path,
            input_name=wake_cfg.input_name,
            output_name=wake_cfg.output_name,
            features_cfg=cfg.features,
            audio_cfg=cfg.audio,
            get_audio_samples=mic.get_last_samples,
            metrics=metrics,
            threshold=wake_cfg.threshold,
            consecutive_hits=wake_cfg.consecutive_hits,
            cooldown_ms=wake_cfg.cooldown_ms,
        )

    if wake_cfg.type == "openwakeword":
        return OpenWakeWordDetector(
            models=wake_cfg.openwakeword.models,
            features_cfg=cfg.features,
            audio_cfg=cfg.audio,
            get_audio_samples=mic.get_last_samples,
            metrics=metrics,
            threshold=wake_cfg.threshold,
            consecutive_hits=wake_cfg.consecutive_hits,
            cooldown_ms=wake_cfg.cooldown_ms,
        )

    raise ValueError(f"Unknown wakeword type: {wake_cfg.type}")

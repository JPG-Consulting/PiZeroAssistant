from assistant.wakeword.metrics import WakeWordMetrics
from assistant.wakeword.detector import SimulatedWakeWordDetector


def create_wake_detector(cfg, metrics: WakeWordMetrics):
    wake_cfg = cfg.wakeword

    if wake_cfg.type == "simulated":
        sim = wake_cfg.simulated
        return SimulatedWakeWordDetector(
            metrics=metrics,
            cooldown_sec=sim.cooldown_sec,
            trigger_probability=sim.trigger_probability,
        )

    raise ValueError(f"Unknown wakeword type: {wake_cfg.type}")

from __future__ import annotations

import time
import random

from assistant.wakeword.metrics import WakeWordMetrics


class SimulatedWakeWordDetector:
    """
    Fake wake-word detector used to test pipeline and metrics.
    """

    def __init__(
        self,
        *,
        metrics: WakeWordMetrics,
        cooldown_sec: float = 5.0,
        trigger_probability: float = 0.02,
    ):
        self.metrics = metrics
        self.cooldown_sec = cooldown_sec
        self.trigger_probability = trigger_probability

        self._last_trigger = 0.0

    def process_frame(self, vad_speech: bool) -> bool:
        """
        Called on every audio frame.
        Returns True if wake-word is triggered.
        """
        now = time.time()

        # Only consider triggering during speech
        if not vad_speech:
            return False

        # Cooldown enforcement
        if now - self._last_trigger < self.cooldown_sec:
            self.metrics.on_suppressed()
            return False

        # Simulated detection
        if random.random() < self.trigger_probability:
            self._last_trigger = now
            self.metrics.on_trigger()
            print("[wake] simulated wake-word triggered")
            return True

        return False

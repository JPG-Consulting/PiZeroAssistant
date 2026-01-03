from __future__ import annotations

import importlib.util
import time
from typing import Sequence

import numpy as np

from assistant.wakeword.metrics import WakeWordMetrics


class OpenWakeWordDetector:
    def __init__(
        self,
        *,
        models: Sequence[str],
        audio_cfg,
        features_cfg,
        get_audio_samples,  # callable: (num_samples) -> np.int16
        metrics: WakeWordMetrics,
        threshold: float,
        consecutive_hits: int,
        cooldown_ms: int,
    ):
        if importlib.util.find_spec("openwakeword") is None:
            raise ImportError(
                "The openwakeword package is required for wakeword.type=openwakeword. "
                "Install it with `pip install openwakeword`."
            )

        from openwakeword.model import Model

        self.metrics = metrics
        self.threshold = threshold
        self.consecutive_hits = consecutive_hits
        self.cooldown_sec = cooldown_ms / 1000.0

        self._get_audio_samples = get_audio_samples
        self._hit_count = 0
        self._last_trigger = 0.0

        self._window_samples = int(audio_cfg.sample_rate * features_cfg.clip_seconds)
        if self._window_samples <= 0:
            raise ValueError("features.clip_seconds must be > 0 for OpenWakeWord")

        model_kwargs = {}
        if models:
            model_kwargs["wakeword_models"] = list(models)

        self.model = Model(**model_kwargs)

    # -----------------------------

    def process_frame(self, vad_speech: bool) -> bool:
        if not vad_speech:
            self._hit_count = 0
            return False

        now = time.time()
        if now - self._last_trigger < self.cooldown_sec:
            self.metrics.on_suppressed()
            return False

        samples = self._get_audio_samples(self._window_samples)
        if samples is None:
            return False

        audio = samples.astype(np.float32) / 32768.0
        probs = self.model.predict(audio)
        if not probs:
            return False

        max_prob = max(float(v) for v in probs.values())
        if max_prob >= self.threshold:
            self._hit_count += 1
            if self._hit_count >= self.consecutive_hits:
                self._last_trigger = now
                self._hit_count = 0
                self.metrics.on_trigger()
                return True
        else:
            self._hit_count = 0

        return False

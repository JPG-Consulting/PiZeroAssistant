from __future__ import annotations

import time
import numpy as np
import onnxruntime as ort

from assistant.dsp import LogMelExtractor
from assistant.wakeword.metrics import WakeWordMetrics


class OnnxWakeWordDetector:
    def __init__(
        self,
        *,
        model_path: str,
        input_name: str,
        output_name: str,
        features_cfg,
        audio_cfg,
        get_audio_samples,           # callable: (num_samples) -> np.int16
        metrics: WakeWordMetrics,
        threshold: float,
        consecutive_hits: int,
        cooldown_ms: int,
    ):
        self.metrics = metrics
        self.threshold = threshold
        self.consecutive_hits = consecutive_hits
        self.cooldown_sec = cooldown_ms / 1000.0

        self._get_audio_samples = get_audio_samples

        self._hit_count = 0
        self._last_trigger = 0.0

        # Feature extractor
        self.feats = LogMelExtractor(
            sr=audio_cfg.sample_rate,
            n_fft=features_cfg.n_fft,
            win_ms=features_cfg.win_ms,
            hop_ms=features_cfg.hop_ms,
            n_mels=features_cfg.n_mels,
            fmin=features_cfg.fmin,
            fmax=features_cfg.fmax,
            log_eps=features_cfg.log_eps,
            clip_seconds=features_cfg.clip_seconds,
        )

        # ONNX session (CPU only)
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = input_name
        self.output_name = output_name

    # -----------------------------

    def process_frame(self, vad_speech: bool) -> bool:
        if not vad_speech:
            self._hit_count = 0
            return False

        now = time.time()
        if now - self._last_trigger < self.cooldown_sec:
            self.metrics.on_suppressed()
            return False

        samples = self._get_audio_samples(self.feats.num_samples)
        if samples is None:
            return False

        logmel = self.feats.extract(samples)  # (1, n_mels, frames)

        p_wake = self._infer(logmel)

        if p_wake >= self.threshold:
            self._hit_count += 1
            if self._hit_count >= self.consecutive_hits:
                self._last_trigger = now
                self._hit_count = 0
                self.metrics.on_trigger()
                return True
        else:
            self._hit_count = 0

        return False

    # -----------------------------

    def _infer(self, logmel: np.ndarray) -> float:
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: logmel.astype(np.float32)},
        )[0]

        # Normalize output
        if outputs.ndim == 2 and outputs.shape[1] == 2:
            # logits or probs
            probs = self._softmax(outputs)[0]
            return float(probs[1])
        else:
            return float(outputs.squeeze())

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

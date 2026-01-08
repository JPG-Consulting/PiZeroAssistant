from __future__ import annotations

import logging
import time
import numpy as np
import onnxruntime as ort

from assistant.dsp import LogMelExtractor
from assistant.wakeword.metrics import WakeWordMetrics


def runtime_preprocess(samples: np.ndarray) -> np.ndarray:
    samples = samples.astype(np.float32)
    samples *= 2.5
    samples = np.tanh(samples / 20000.0) * 20000.0
    return samples.astype(np.int16)


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
        self._sample_rate = audio_cfg.sample_rate
        self._log = logging.getLogger("assistant.wakeword.onnx")

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

        print("ONNX inputs :", [(i.name, i.shape, i.type) for i in self.session.get_inputs()])
        print("ONNX outputs:", [(o.name, o.shape, o.type) for o in self.session.get_outputs()])


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

        inference_start_wall = time.time()
        inference_start_mono = time.monotonic()
        self._log.debug(
            "[WAKE_TRACE] inference_start wall=%.6f mono=%.6f",
            inference_start_wall,
            inference_start_mono,
        )

        samples = self._get_audio_samples(self.feats.num_samples)
        window_end_wall = time.time()
        window_end_mono = time.monotonic()
        window_duration = self.feats.num_samples / float(self._sample_rate)
        window_start_wall = window_end_wall - window_duration
        window_start_mono = window_end_mono - window_duration
        self._log.debug(
            "[WAKE_TRACE] audio_window wall=%.6f..%.6f mono=%.6f..%.6f samples=%d",
            window_start_wall,
            window_end_wall,
            window_start_mono,
            window_end_mono,
            self.feats.num_samples,
        )

        # -------------------------------------------------
        # Software gain control (STABLE, deterministic)
        # -------------------------------------------------
        samples = runtime_preprocess(samples)

        print(
            "samples stats:",
            "dtype=", samples.dtype,
            "min=", samples.min(),
            "max=", samples.max(),
            "mean=", samples.mean(),
            "std=", samples.std(),
        )

        if samples is None:
            return False

        logmel = self.feats.extract(samples)  # (1, n_mels, frames)

        print("logmel mean:", logmel.mean(), "std:", logmel.std())

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
        x = np.ascontiguousarray(logmel, dtype=np.float32)

        # Ensure shape is (B, C, n_mels, frames)
        if x.ndim == 3:
            x = x[:, None, :, :]   # -> (1, 1, n_mels, frames)

        prob = self.session.run(
            [self.output_name],
            {self.input_name: x},
        )[0]

        prob = float(prob.squeeze())

        # 🔍 DEBUG LOG
        print(f"[WAKE] prob={prob:.6f}")

        return prob

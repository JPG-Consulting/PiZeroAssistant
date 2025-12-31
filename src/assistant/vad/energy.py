from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass


@dataclass
class EnergyVAD:
    rms_threshold: float
    hangover_ms: int
    block_ms: int
    start_frames: int = 1

    noise_alpha: float = 0.02
    noise_factor: float = 2.5

    def __post_init__(self) -> None:
        if self.start_frames < 1:
            raise ValueError("start_frames must be >= 1")

        self._speech = False
        self._above_count = 0
        self._speech_until = 0.0

        self._noise_floor = None  # learned dynamically

    def is_speech(self, pcm16: bytes, sample_rate: int = 16000) -> bool:
        """
        Adaptive energy-based VAD.

        pcm16: raw PCM16 bytes
        """
        # PCM16 → float [-1, 1]
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

        # Remove DC offset (important for I²S mics)
        x = x - np.mean(x)

        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        now = time.time()

        # Initialize noise floor conservatively
        if self._noise_floor is None:
            self._noise_floor = rms

        # Update noise floor only when NOT in speech
        if not self._speech:
            self._noise_floor = (
                (1.0 - self.noise_alpha) * self._noise_floor
                + self.noise_alpha * rms
            )

        adaptive_threshold = max(
            self.rms_threshold,
            self._noise_floor * self.noise_factor,
        )

        if rms >= adaptive_threshold:
            self._above_count += 1
        else:
            self._above_count = 0

        # Enter SPEECH after N consecutive frames
        if not self._speech and self._above_count >= self.start_frames:
            self._speech = True

        if self._speech:
            if rms >= adaptive_threshold:
                self._speech_until = now + (self.hangover_ms / 1000.0)
            elif now > self._speech_until:
                self._speech = False
                self._above_count = 0

        return self._speech

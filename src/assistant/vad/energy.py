from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np


@dataclass
class EnergyVAD:
    """
    Very small fallback VAD based on RMS energy with hangover.
    Not as good as WebRTC VAD but works fully offline with minimal deps.
    """
    rms_threshold: float = 0.015
    hangover_ms: int = 200

    _speech_until: float = 0.0

    def is_speech(self, pcm16: bytes, sample_rate: int = 16000) -> bool:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        now = time.time()

        if rms >= self.rms_threshold:
            self._speech_until = now + (self.hangover_ms / 1000.0)
            return True

        return now <= self._speech_until

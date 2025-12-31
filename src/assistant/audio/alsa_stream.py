from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import alsaaudio


@dataclass
class AudioFrame:
    pcm16: bytes
    timestamp: float


class AlsaMicStream:
    """
    Pure ALSA microphone stream using pyalsaaudio.
    No PortAudio, no sounddevice.
    """

    def __init__(
        self,
        sample_rate: int,
        block_ms: int,
        device: Optional[str] = None,
        channels: int = 1,
        ring_seconds: float = 3.0,
    ):
        if channels != 1:
            raise ValueError("Only mono audio is supported")

        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.channels = channels
        self.device = device or "default"

        self.block_samples = int(sample_rate * block_ms / 1000)
        self.block_bytes = self.block_samples * 2  # int16

        # Ring buffer
        self._ring_size = int(sample_rate * ring_seconds)
        self._ring = np.zeros((self._ring_size,), dtype=np.int16)
        self._ring_wpos = 0

        self._frames: list[AudioFrame] = []
        self._running = False

        # ALSA PCM
        self._pcm = alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE,
            mode=alsaaudio.PCM_NORMAL,
            device=self.device,
        )
        self._pcm.setchannels(1)
        self._pcm.setrate(self.sample_rate)
        self._pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self._pcm.setperiodsize(self.block_samples)

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def close(self):
        self._pcm.close()

    def poll(self):
        """
        Poll ALSA for new audio.
        Call this frequently from the main loop.
        """
        if not self._running:
            return

        length, data = self._pcm.read()
        if length <= 0:
            return

        ts = time.time()
        self._frames.append(AudioFrame(pcm16=data, timestamp=ts))

        samples = np.frombuffer(data, dtype=np.int16)
        n = len(samples)

        end = self._ring_wpos + n
        if end <= self._ring_size:
            self._ring[self._ring_wpos:end] = samples
        else:
            first = self._ring_size - self._ring_wpos
            self._ring[self._ring_wpos:] = samples[:first]
            self._ring[: end % self._ring_size] = samples[first:]

        self._ring_wpos = end % self._ring_size

    def pop_frame(self) -> Optional[AudioFrame]:
        if not self._frames:
            return None
        return self._frames.pop(0)

    def get_last_samples(self, num_samples: int) -> np.ndarray:
        if num_samples > self._ring_size:
            raise ValueError("Requested more samples than ring buffer size")

        start = (self._ring_wpos - num_samples) % self._ring_size
        if start < self._ring_wpos:
            out = self._ring[start:self._ring_wpos].copy()
        else:
            out = np.concatenate(
                [self._ring[start:], self._ring[:self._ring_wpos]]
            ).copy()

        return out

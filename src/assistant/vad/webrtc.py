from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import webrtcvad  # type: ignore
except Exception:  # pragma: no cover
    webrtcvad = None


@dataclass
class WebRtcVAD:
    aggressiveness: int = 2
    sample_rate: int = 16000
    frame_ms: int = 20

    def __post_init__(self):
        if webrtcvad is None:
            raise RuntimeError(
                "webrtcvad is not installed. Install it or switch vad.type to 'energy'."
            )
        if self.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("WebRTC VAD supports 8/16/32/48 kHz.")
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD frame_ms must be 10, 20, or 30.")

        self._vad = webrtcvad.Vad(int(self.aggressiveness))

    def is_speech(self, pcm16: bytes) -> bool:
        # pcm16 must be mono 16-bit little-endian PCM matching sample_rate and frame_ms length
        return bool(self._vad.is_speech(pcm16, self.sample_rate))

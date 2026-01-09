"""Real-time audio capture."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import sounddevice as sd

from voiceassistant.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AudioFrame:
    pcm: bytes
    sample_rate_hz: int


class AudioCaptureThread:
    """Captures audio frames and distributes them to consumer queues."""

    def __init__(
        self,
        sample_rate_hz: int,
        channels: int,
        frame_duration_ms: int,
        device: Optional[str],
        wakeword_queue: queue.Queue[AudioFrame],
        record_queue: queue.Queue[AudioFrame],
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.device = device
        self._wakeword_queue = wakeword_queue
        self._record_queue = record_queue
        self._stream: Optional[sd.InputStream] = None
        self._running = threading.Event()
        self._dropped_wakeword = 0
        self._dropped_record = 0

    def _frame_samples(self) -> int:
        return int(self.sample_rate_hz * self.frame_duration_ms / 1000)

    def _put_drop(self, q: queue.Queue[AudioFrame], frame: AudioFrame, kind: str) -> None:
        try:
            q.put_nowait(frame)
        except queue.Full:
            if kind == "wakeword":
                self._dropped_wakeword += 1
            else:
                self._dropped_record += 1

    def _callback(self, indata, frames, time, status) -> None:  # pragma: no cover - realtime
        if status:
            logger.warning("Audio capture status: %s", status)
        if not self._running.is_set():
            return
        pcm = indata.tobytes()
        frame = AudioFrame(pcm=pcm, sample_rate_hz=self.sample_rate_hz)
        self._put_drop(self._wakeword_queue, frame, "wakeword")
        self._put_drop(self._record_queue, frame, "record")

    def start(self) -> None:
        if self._stream:
            return
        self._running.set()
        frame_samples = self._frame_samples()
        logger.info("Starting audio capture at %s Hz", self.sample_rate_hz)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate_hz,
            channels=self.channels,
            blocksize=frame_samples,
            dtype="int16",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running.clear()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._dropped_wakeword or self._dropped_record:
            logger.warning(
                "Dropped frames: wakeword=%s record=%s",
                self._dropped_wakeword,
                self._dropped_record,
            )

"""Voice recorder with VAD-based stop."""

from __future__ import annotations

import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

import webrtcvad

from voiceassistant.audio.capture import AudioFrame
from voiceassistant.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RecordingResult:
    pcm: bytes
    sample_rate_hz: int
    channels: int


class Recorder:
    """Consumes audio frames and records a bounded utterance."""

    def __init__(
        self,
        frame_duration_ms: int,
        sample_rate_hz: int,
        channels: int,
        vad_mode: int,
        max_record_seconds: int,
        record_silence_ms: int,
        record_queue: queue.Queue[AudioFrame],
    ) -> None:
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.max_record_seconds = max_record_seconds
        self.record_silence_ms = record_silence_ms
        self._vad = webrtcvad.Vad(vad_mode)
        self._queue = record_queue
        self._recording = threading.Event()
        self._result: Optional[RecordingResult] = None
        self._thread = threading.Thread(target=self._run, name="Recorder", daemon=True)
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def begin_recording(self, pre_roll: bytes) -> None:
        with self._lock:
            self._result = RecordingResult(
                pcm=pre_roll,
                sample_rate_hz=self.sample_rate_hz,
                channels=self.channels,
            )
        self._recording.set()

    def wait_for_result(self, timeout_s: float) -> Optional[RecordingResult]:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if not self._recording.is_set():
                with self._lock:
                    return self._result
            time.sleep(0.01)
        return None

    def _run(self) -> None:  # pragma: no cover - loop
        silence_ms = 0
        max_frames = int(self.max_record_seconds * 1000 / self.frame_duration_ms)
        frames_seen = 0
        while True:
            frame = self._queue.get()
            if not self._recording.is_set():
                continue
            if frame.sample_rate_hz != self.sample_rate_hz:
                logger.warning("Discarding frame with mismatched sample rate")
                continue
            is_speech = self._vad.is_speech(frame.pcm, self.sample_rate_hz)
            with self._lock:
                if self._result:
                    self._result.pcm += frame.pcm
            frames_seen += 1
            if is_speech:
                silence_ms = 0
            else:
                silence_ms += self.frame_duration_ms
            if silence_ms >= self.record_silence_ms or frames_seen >= max_frames:
                self._recording.clear()
                silence_ms = 0
                frames_seen = 0

    def save_wav(self, result: RecordingResult, path: str) -> None:
        with wave.open(path, "wb") as handle:
            handle.setnchannels(result.channels)
            handle.setsampwidth(2)
            handle.setframerate(result.sample_rate_hz)
            handle.writeframes(result.pcm)
        logger.info("Saved recording to %s", path)

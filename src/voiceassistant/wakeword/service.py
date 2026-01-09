"""Wakeword detection service."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import webrtcvad

from voiceassistant.audio.capture import AudioFrame
from voiceassistant.logging_config import get_logger
from voiceassistant.utils.ring_buffer import ByteRingBuffer

logger = get_logger(__name__)

try:
    from openwakeword.model import Model
except ImportError:  # pragma: no cover - optional dependency
    Model = None


@dataclass(frozen=True)
class WakewordEvent:
    timestamp: float
    pre_roll_pcm: bytes


class WakewordService:
    """Dedicated wakeword detector with VAD and pre-roll buffer."""

    def __init__(
        self,
        sample_rate_hz: int,
        frame_duration_ms: int,
        inference_window_ms: int,
        model_path: str,
        score_threshold: float,
        wakeword_cooldown_ms: int,
        pre_roll_ms: int,
        vad_mode: int,
        input_queue: queue.Queue[AudioFrame],
        event_queue: queue.Queue[WakewordEvent],
    ) -> None:
        if Model is None:
            raise RuntimeError("openwakeword is not installed")
        self.sample_rate_hz = sample_rate_hz
        self.frame_duration_ms = frame_duration_ms
        self.inference_window_ms = inference_window_ms
        self.score_threshold = score_threshold
        self.wakeword_cooldown_ms = wakeword_cooldown_ms
        self._input_queue = input_queue
        self._event_queue = event_queue
        self._vad = webrtcvad.Vad(vad_mode)
        self._model = Model(wakeword_models=[model_path])
        self._thread = threading.Thread(target=self._run, name="Wakeword", daemon=True)
        self._running = threading.Event()
        self._last_wake_ts = 0.0

        frame_bytes = int(sample_rate_hz * frame_duration_ms / 1000) * 2
        frame_count = max(1, int(pre_roll_ms / frame_duration_ms))
        self._pre_roll = ByteRingBuffer(frame_bytes=frame_bytes, frame_count=frame_count)

    def start(self) -> None:
        if self._thread.is_alive():
            return
        logger.info("Starting wakeword service")
        self._running.set()
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()

    def _run(self) -> None:  # pragma: no cover - realtime
        buffer = bytearray()
        frame_bytes = int(self.sample_rate_hz * self.frame_duration_ms / 1000) * 2
        while self._running.is_set():
            try:
                frame = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if len(frame.pcm) != frame_bytes:
                logger.warning("Dropping invalid frame length")
                continue

            self._pre_roll.append(frame.pcm)

            vad_frame = bytes(frame.pcm)
            kws_frame = bytes(frame.pcm)
            _ = self._vad.is_speech(vad_frame, self.sample_rate_hz)

            buffer.extend(kws_frame)
            target_bytes = int(self.sample_rate_hz * self.inference_window_ms / 1000) * 2
            if len(buffer) < target_bytes:
                continue
            window = bytes(buffer[:target_bytes])
            buffer = buffer[target_bytes:]

            scores = self._model.predict(window)
            if not scores:
                continue
            score = max(scores.values())
            if score < self.score_threshold:
                continue
            now = time.monotonic()
            if (now - self._last_wake_ts) * 1000 < self.wakeword_cooldown_ms:
                continue
            self._last_wake_ts = now
            event = WakewordEvent(timestamp=now, pre_roll_pcm=self._pre_roll.snapshot())
            try:
                self._event_queue.put_nowait(event)
                logger.info("Wakeword detected")
            except queue.Full:
                logger.warning("Wakeword event dropped: queue full")

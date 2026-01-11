"""Wakeword detection service."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
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
        model_file = Path(model_path)
        if not model_file.is_file():
            logger.error("Wakeword model file not found: %s", model_path)
            raise FileNotFoundError(f"Wakeword model file not found: {model_path}")
        model_suffix = model_file.suffix.lower()
        if model_suffix not in {".onnx", ".tflite"}:
            logger.error(
                "Unsupported wakeword model format: %s (expected .onnx or .tflite)",
                model_suffix,
            )
            raise ValueError(
                f"Unsupported wakeword model format: {model_suffix} (expected .onnx or .tflite)"
            )
        backend = "onnx" if model_suffix == ".onnx" else "tflite"
        logger.info(
            "Wakeword model resolved: path=%s suffix=%s backend=%s",
            model_path,
            model_suffix,
            backend,
        )
        if (backend == "onnx" and model_suffix != ".onnx") or (
            backend == "tflite" and model_suffix != ".tflite"
        ):
            raise ValueError(
                "Wakeword configuration error:\n"
                f"backend='{backend}' but model='{model_path}'\n\n"
                "The wakeword backend and model format must match.\n"
                "Use a .tflite model for the TFLite backend or switch the backend to ONNX."
            )
        self.sample_rate_hz = sample_rate_hz
        self.frame_duration_ms = frame_duration_ms
        self.inference_window_ms = inference_window_ms
        self.score_threshold = score_threshold
        self.wakeword_cooldown_ms = wakeword_cooldown_ms
        self._input_queue = input_queue
        self._event_queue = event_queue
        self._vad = webrtcvad.Vad(vad_mode)
        self._model = Model(wakeword_models=[model_path], inference_framework=backend)
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

    @staticmethod
    def _describe_audio_input(x: Any) -> str:
        if isinstance(x, np.ndarray):
            return f"type=ndarray dtype={x.dtype} shape={x.shape}"
        try:
            length = len(x)
        except TypeError:
            length = "unknown"
        return f"type={type(x).__name__} len={length}"

    @staticmethod
    def _to_int16_mono_array(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            array = x
        elif isinstance(x, (bytes, bytearray, memoryview)):
            array = np.frombuffer(x, dtype=np.int16)
        else:
            raise ValueError(
                "Wakeword audio input must be a numpy array or raw PCM bytes"
            )

        if array.dtype != np.int16:
            array = array.astype(np.int16, copy=False)

        if array.ndim != 1:
            array = np.squeeze(array)
            if array.ndim != 1:
                raise ValueError(
                    "Wakeword audio input must be a 1-D int16 mono array"
                )

        return array

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

            try:
                audio = self._to_int16_mono_array(window)
                scores = self._model.predict(audio)
            except ValueError as exc:
                logger.warning(
                    "Wakeword input validation failed: %s (%s)",
                    exc,
                    self._describe_audio_input(window),
                )
                continue
            except Exception:
                logger.exception(
                    "Wakeword inference failed (%s)",
                    self._describe_audio_input(window),
                )
                continue
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

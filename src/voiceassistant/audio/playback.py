"""Audio playback controller with barge-in support."""

from __future__ import annotations

import io
import threading
import wave
from dataclasses import dataclass
from typing import Optional

import sounddevice as sd

from voiceassistant.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PlaybackRequest:
    wav_bytes: bytes


class PlaybackController:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def play(self, request: PlaybackRequest) -> None:
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._playback, args=(request,), name="Playback", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def is_playing(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _playback(self, request: PlaybackRequest) -> None:  # pragma: no cover - realtime
        try:
            with wave.open(io.BytesIO(request.wav_bytes), "rb") as handle:
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
                dtype = "int16" if handle.getsampwidth() == 2 else "int32"
                stream = sd.RawOutputStream(
                    samplerate=sample_rate,
                    channels=channels,
                    dtype=dtype,
                )
                with stream:
                    while not self._stop_event.is_set():
                        chunk = handle.readframes(1024)
                        if not chunk:
                            break
                        stream.write(chunk)
        except wave.Error:
            logger.exception("Failed to play audio: invalid WAV data")
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected playback error")

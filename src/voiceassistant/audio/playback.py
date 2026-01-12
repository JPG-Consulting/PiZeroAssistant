"""Audio playback controller with barge-in support."""

from __future__ import annotations

import io
import threading
import wave
import time
from dataclasses import dataclass
from typing import Optional

import sounddevice as sd

from voiceassistant.audio.stream import AudioStream, BytesAudioStream
from voiceassistant.audio.ffmpeg_decoder import FFMpegDecodeError, decode_to_pcm_stream
from voiceassistant.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PlaybackRequest:
    wav_bytes: Optional[bytes] = None
    audio: Optional[AudioStream] = None


class PlaybackController:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._last_elapsed_time: Optional[float] = None
        self._last_stop_reason = "unknown"
        self._expected_duration: Optional[float] = None

    def play(self, request: PlaybackRequest) -> None:
        self.stop()
        self._stop_event.clear()
        expected_duration = self._estimate_expected_duration(request)
        with self._lock:
            self._start_time = time.monotonic()
            self._last_elapsed_time = None
            self._last_stop_reason = "unknown"
            self._expected_duration = expected_duration
        if expected_duration is not None:
            logger.debug("Playback started (expected=%.2fs)", expected_duration)
        else:
            logger.debug("Playback started")
        self._thread = threading.Thread(
            target=self._playback, args=(request,), name="Playback", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        should_log = self.is_playing()
        elapsed = None
        with self._lock:
            self._stop_event.set()
            if self._start_time is not None:
                elapsed = time.monotonic() - self._start_time
                self._last_elapsed_time = elapsed
                self._last_stop_reason = "explicit_stop"
                self._expected_duration = None
        if should_log and elapsed is not None:
            logger.debug("Playback stopped explicitly (elapsed=%.2fs)", elapsed)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def is_playing(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def get_last_stop_reason(self) -> str:
        with self._lock:
            return self._last_stop_reason

    def get_last_elapsed_time(self) -> Optional[float]:
        with self._lock:
            return self._last_elapsed_time

    def set_stop_reason(self, reason: str) -> None:
        with self._lock:
            self._last_stop_reason = reason

    def _playback(self, request: PlaybackRequest) -> None:  # pragma: no cover - realtime
        if request.audio is None and request.wav_bytes is None:
            logger.warning("Playback request contained no audio data")
            return
        self._play_audio_stream(request)

    def _play_audio_stream(self, request: PlaybackRequest) -> None:
        decoder = None
        audio: Optional[AudioStream] = None
        try:
            audio = self._coerce_audio_stream(request)
            if audio is None:
                return
            logger.debug(
                "Playback audio format=%s sample_rate_hz=%s channels=%s",
                audio.format,
                audio.sample_rate_hz,
                audio.channels,
            )
            stream = sd.RawOutputStream(
                samplerate=audio.sample_rate_hz,
                channels=audio.channels,
                dtype="int16",
            )
            with stream:
                decoder = decode_to_pcm_stream(audio)
                total_bytes = 0
                chunk_count = 0
                stopped = False
                for chunk in decoder:
                    # Stop latency is bounded by the decoder select timeout (~100ms);
                    # this trade-off keeps barge-in responsive without busy-looping.
                    if self._stop_event.is_set():
                        logger.debug("Playback stop event set; halting stream write")
                        stopped = True
                        break
                    if not chunk:
                        continue
                    stream.write(chunk)
                    total_bytes += len(chunk)
                    chunk_count += 1
                if stopped:
                    logger.debug(
                        "Playback stopped after %d chunks (%d bytes)",
                        chunk_count,
                        total_bytes,
                    )
                else:
                    logger.debug(
                        "Playback decoder completed after %d chunks (%d bytes)",
                        chunk_count,
                        total_bytes,
                    )
                    elapsed = None
                    with self._lock:
                        if self._start_time is not None:
                            elapsed = time.monotonic() - self._start_time
                            self._last_elapsed_time = elapsed
                            if self._last_stop_reason == "unknown":
                                self._last_stop_reason = "normal_end"
                            expected = self._expected_duration
                            self._expected_duration = None
                            stop_reason = self._last_stop_reason
                    if elapsed is not None:
                        logger.debug("Playback finished normally (elapsed=%.2fs)", elapsed)
                        if (
                            expected is not None
                            and stop_reason == "normal_end"
                            and elapsed < expected * 0.85
                        ):
                            logger.warning(
                                "Truncated playback detected: expected=%.2fs actual=%.2fs",
                                expected,
                                elapsed,
                            )
                    # Explicit stop/close ordering helps drain buffered audio on EOF.
                    stream.stop()
                    stream.close()
        except FFMpegDecodeError as exc:
            logger.exception("Failed to decode audio for playback: %s", exc)
        except wave.Error:
            logger.exception("Failed to play audio: invalid WAV data")
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected playback error")
        finally:
            # Close the decoder before the source stream to allow graceful shutdown ordering.
            if decoder is not None:
                decoder.close()
            if audio is not None and hasattr(audio, "close"):
                audio.close()

    def _coerce_audio_stream(self, request: PlaybackRequest) -> Optional[AudioStream]:
        if request.audio is not None:
            return request.audio
        if request.wav_bytes is None:
            return None
        try:
            with wave.open(io.BytesIO(request.wav_bytes), "rb") as handle:
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
        except wave.Error as exc:
            raise wave.Error("Failed to parse WAV metadata") from exc
        return BytesAudioStream(
            data=request.wav_bytes,
            format="wav",
            sample_rate_hz=sample_rate,
            channels=channels,
        )

    def _estimate_expected_duration(self, request: PlaybackRequest) -> Optional[float]:
        if request.wav_bytes is None:
            return None
        try:
            with wave.open(io.BytesIO(request.wav_bytes), "rb") as handle:
                frames = handle.getnframes()
                sample_rate_hz = handle.getframerate()
        except wave.Error:
            return None
        if sample_rate_hz <= 0:
            return None
        return frames / sample_rate_hz

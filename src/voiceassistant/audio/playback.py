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
    provider_name: Optional[str] = None


class PlaybackController:
    """Playback controller enforcing audible completion before reporting `normal_end` (see DEVELOPMENT.md)."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._request_time: Optional[float] = None
        self._last_elapsed_time: Optional[float] = None
        self._last_stop_reason = "unknown"
        self._expected_audio_duration: Optional[float] = None

    def play(self, request: PlaybackRequest) -> None:
        self.stop()
        self._stop_event.clear()
        request_time = time.monotonic()
        expected_audio_duration = self._estimate_expected_duration(request)
        with self._lock:
            self._request_time = request_time
            self._start_time = None
            self._last_elapsed_time = None
            self._last_stop_reason = "unknown"
            self._expected_audio_duration = expected_audio_duration
        if expected_audio_duration is not None:
            logger.debug("Playback queued (expected_audio=%.2fs)", expected_audio_duration)
        else:
            logger.debug("Playback queued")
        self._thread = threading.Thread(
            target=self._playback, args=(request, request_time), name="Playback", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        should_log = self.is_playing()
        elapsed = None
        with self._lock:
            logger.debug(
                "Playback stop requested (is_playing=%s stop_reason=%s)",
                should_log,
                self._last_stop_reason,
            )
            self._stop_event.set()
            if self._start_time is not None:
                elapsed = time.monotonic() - self._start_time
                self._last_elapsed_time = elapsed
                if self._last_stop_reason == "unknown":
                    self._last_stop_reason = "explicit_stop"
                self._expected_audio_duration = None
            self._request_time = None
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

    def _playback(  # pragma: no cover - realtime
        self,
        request: PlaybackRequest,
        request_time: float,
    ) -> None:
        if request.audio is None and request.wav_bytes is None:
            logger.warning("Playback request contained no audio data")
            return
        self._play_audio_stream(request, request_time)

    def _play_audio_stream(self, request: PlaybackRequest, request_time: float) -> None:
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
                started = False
                ttfs_logged = False
                last_write_time = None
                last_chunk_size = None
                for chunk in decoder:
                    # Stop latency is bounded by the decoder select timeout (~100ms);
                    # this trade-off keeps barge-in responsive without busy-looping.
                    if self._stop_event.is_set():
                        logger.debug("Playback stop event set; halting stream write")
                        stopped = True
                        break
                    if not chunk:
                        continue
                    if not started:
                        first_audio_time = time.monotonic()
                        with self._lock:
                            self._start_time = first_audio_time
                        if not ttfs_logged:
                            ttfs_ms = (first_audio_time - request_time) * 1000
                            if request.provider_name:
                                logger.debug(
                                    "[TTS] time_to_first_audio_ms=%.0f provider=%s",
                                    ttfs_ms,
                                    request.provider_name,
                                )
                            else:
                                logger.debug("[TTS] time_to_first_audio_ms=%.0f", ttfs_ms)
                            ttfs_logged = True
                        started = True
                    # Playback starts on first available audio chunk to minimize latency.
                    stream.write(chunk)
                    last_write_time = time.monotonic()
                    last_chunk_size = len(chunk)
                    total_bytes += len(chunk)
                    chunk_count += 1
                last_write_time_s = (
                    f"{last_write_time:.6f}" if last_write_time is not None else "None"
                )
                logger.debug(
                    "Playback decode loop exit (stop_event=%s chunk_count=%d total_bytes=%d last_chunk_size=%s last_write_time=%s)",
                    self._stop_event.is_set(),
                    chunk_count,
                    total_bytes,
                    last_chunk_size,
                    last_write_time_s,
                )
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
                    # Drain is part of normal completion; audible playback ends after drain.
                    drain_duration_ms = self._drain_output(stream)
                    elapsed = None
                    expected_audio = None
                    stop_reason = None
                    with self._lock:
                        if self._start_time is not None:
                            elapsed = time.monotonic() - self._start_time
                            self._last_elapsed_time = elapsed
                            if self._last_stop_reason == "unknown":
                                self._last_stop_reason = "normal_end"
                            expected_audio = self._expected_audio_duration
                            self._expected_audio_duration = None
                            self._request_time = None
                            stop_reason = self._last_stop_reason
                    if not self._stop_event.is_set() and stop_reason == "normal_end":
                        grace_delay_ms = 50
                        logger.debug(
                            "Applying hardware buffer grace delay (duration_ms=%d)",
                            grace_delay_ms,
                        )
                        time.sleep(grace_delay_ms / 1000)
                    if elapsed is not None:
                        logger.debug("Playback finished normally (elapsed=%.2fs)", elapsed)
                        if expected_audio is not None:
                            logger.debug(
                                "Playback duration comparison (expected=%.2fs elapsed=%.2fs drain_ms=%.1f)",
                                expected_audio,
                                elapsed,
                                drain_duration_ms,
                            )
                        if (
                            expected_audio is not None
                            and stop_reason == "normal_end"
                            and elapsed < expected_audio * 0.85
                        ):
                            logger.warning(
                                "Truncated playback detected: expected=%.2fs actual=%.2fs",
                                expected_audio,
                                elapsed,
                            )
                try:
                    if stopped:
                        abort = getattr(stream, "abort", None)
                        if callable(abort):
                            logger.debug("Aborting audio stream after stop event")
                            abort()
                        else:
                            # stop() is reserved for non-normal termination (barge-in/explicit stop).
                            logger.debug("Stopping audio stream after stop event")
                            stream.stop()
                    else:
                        # On normal completion, avoid stop(); allow natural closure for audible completion.
                        logger.debug("Allowing audio stream to close naturally after drain")
                finally:
                    if stopped:
                        logger.debug("Closing audio stream after stop event")
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

    def _drain_output(self, stream: sd.RawOutputStream) -> float:
        if self._stop_event.is_set():
            with self._lock:
                stop_reason = self._last_stop_reason or "unknown"
            logger.debug(
                "Skipping audio drain; stop requested (stop_reason=%s)",
                stop_reason,
            )
            return 0.0
        start = time.monotonic()
        drain = getattr(stream, "drain", None)
        if callable(drain):
            logger.debug("Draining audio output device (start=%.6f)", start)
            drain()
        else:
            logger.debug("Draining audio output device via bounded delay (start=%.6f)", start)
            time.sleep(0.02)
        duration_ms = (time.monotonic() - start) * 1000
        logger.debug("Audio output drain completed in %.1f ms", duration_ms)
        return duration_ms

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

"""Streaming audio decoding via ffmpeg."""

from __future__ import annotations

import os
import selectors
import subprocess
import threading
from typing import Iterator

from voiceassistant.audio.stream import AudioStream
from voiceassistant.logging_config import get_logger


class FFMpegDecodeError(RuntimeError):
    """Raised when ffmpeg decoding fails."""


logger = get_logger(__name__)


def decode_to_pcm_stream(audio: AudioStream, *, chunk_size: int = 4096) -> Iterator[bytes]:
    if audio.format == "pcm":
        total_bytes = 0
        frame_size = 2 * audio.channels
        for chunk in audio.iter_chunks():
            total_bytes += len(chunk)
            yield chunk
        remainder = total_bytes % frame_size
        if remainder:
            yield b"\x00" * (frame_size - remainder)
        return

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(audio.sample_rate_hz),
        "-ac",
        str(audio.channels),
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise FFMpegDecodeError(
            "ffmpeg not found. Install it (apt-get install ffmpeg) to play mp3/opus/wav "
            "TTS streams."
        ) from exc

    stop_event = threading.Event()
    writer_errors: list[Exception] = []

    def _writer() -> None:
        try:
            assert proc.stdin is not None
            for chunk in audio.iter_chunks():
                if stop_event.is_set():
                    break
                if not chunk:
                    continue
                try:
                    proc.stdin.write(chunk)
                    proc.stdin.flush()
                except BrokenPipeError:
                    break
        except Exception as exc:  # pragma: no cover - defensive
            writer_errors.append(exc)
        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass

    thread = threading.Thread(target=_writer, name="FFMpegStdin", daemon=True)
    thread.start()

    selector = selectors.DefaultSelector()
    stderr_tail = bytearray()
    total_stdout_bytes = 0
    frame_size = 2 * audio.channels
    stdout_eof = False
    proc_exited = False
    try:
        assert proc.stdout is not None
        assert proc.stderr is not None
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)
        selector.register(proc.stdout, selectors.EVENT_READ)
        selector.register(proc.stderr, selectors.EVENT_READ)
        while True:
            if stop_event.is_set():
                break
            events = selector.select(timeout=0.1)
            if not events:
                if proc.poll() is not None:
                    proc_exited = True
                    break
                continue
            eof = False
            for key, _ in events:
                if key.fileobj is proc.stderr:
                    data = os.read(key.fileobj.fileno(), chunk_size)
                    if data:
                        stderr_tail.extend(data)
                        if len(stderr_tail) > 200:
                            stderr_tail = stderr_tail[-200:]
                    continue
                chunk = os.read(key.fileobj.fileno(), chunk_size)
                if not chunk:
                    stdout_eof = True
                    eof = True
                    break
                total_stdout_bytes += len(chunk)
                yield chunk
            if eof:
                break
        if not stop_event.is_set():
            while True:
                events = selector.select(timeout=0.1)
                if not events:
                    if proc.poll() is not None:
                        proc_exited = True
                        break
                    continue
                for key, _ in events:
                    if key.fileobj is proc.stderr:
                        data = os.read(key.fileobj.fileno(), chunk_size)
                        if data:
                            stderr_tail.extend(data)
                            if len(stderr_tail) > 200:
                                stderr_tail = stderr_tail[-200:]
                        continue
                    chunk = os.read(key.fileobj.fileno(), chunk_size)
                    if not chunk:
                        stdout_eof = True
                        continue
                    total_stdout_bytes += len(chunk)
                    yield chunk
            proc.wait()
            logger.debug(
                "ffmpeg decode completed stdout_bytes=%d stdout_eof=%s proc_exited=%s returncode=%s",
                total_stdout_bytes,
                stdout_eof,
                proc_exited,
                proc.returncode,
            )
            if writer_errors:
                raise FFMpegDecodeError(str(writer_errors[0]))
            if proc.returncode:
                stderr = stderr_tail.decode("utf-8", "replace").strip().replace("\n", " ")
                if len(stderr) > 200:
                    stderr = f"{stderr[:200]}..."
                message = "ffmpeg decode failed"
                if stderr:
                    message = f"{message}: {stderr}"
                raise FFMpegDecodeError(message)
            remainder = total_stdout_bytes % frame_size
            if remainder:
                yield b"\x00" * (frame_size - remainder)
    except GeneratorExit:
        stop_event.set()
        raise
    finally:
        stop_event.set()
        selector.close()
        if proc.poll() is None:
            if stdout_eof and proc_exited:
                proc.wait()
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
        if thread.is_alive():
            thread.join(timeout=1.0)

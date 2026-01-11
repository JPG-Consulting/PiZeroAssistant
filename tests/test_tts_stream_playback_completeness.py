import io
import math
import shutil
import struct
import unittest
import wave
from dataclasses import dataclass
from typing import Iterator, List

from voiceassistant.audio.ffmpeg_decoder import decode_to_pcm_stream


@dataclass(frozen=True)
class FakeAudioStream:
    format: str
    sample_rate_hz: int
    channels: int
    chunks: List[bytes]

    def iter_chunks(self) -> Iterator[bytes]:
        yield from self.chunks


def _build_test_wav_bytes(
    *,
    sample_rate_hz: int = 8000,
    silence_s: float = 0.5,
    tail_s: float = 0.1,
    tail_amplitude: int = 20000,
    tail_samples: int = 200,
) -> bytes:
    silence_frames = int(sample_rate_hz * silence_s)
    tail_frames = int(sample_rate_hz * tail_s)
    total_frames = silence_frames + tail_frames
    samples: List[int] = [0] * total_frames
    for i in range(tail_frames):
        phase = 2 * math.pi * 440.0 * i / sample_rate_hz
        samples[silence_frames + i] = int(0.6 * 32767 * math.sin(phase))
    for i in range(1, tail_samples + 1):
        samples[-i] = tail_amplitude
    pcm_bytes = struct.pack(f"<{len(samples)}h", *samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm_bytes)
    return buffer.getvalue()


class TestTTSStreamPlaybackCompleteness(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not available")
    def test_decoding_preserves_trailing_audio(self) -> None:
        wav_bytes = _build_test_wav_bytes()
        with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
            expected_frames = handle.getnframes()
            expected_pcm_bytes = expected_frames * handle.getnchannels() * 2
        chunks = [wav_bytes[i : i + 513] for i in range(0, len(wav_bytes), 513)]
        audio = FakeAudioStream(
            format="wav",
            sample_rate_hz=8000,
            channels=1,
            chunks=chunks,
        )
        decoded_pcm = b"".join(decode_to_pcm_stream(audio))
        tolerance = 2048
        if len(decoded_pcm) != expected_pcm_bytes:
            print(
                f"Decoded PCM length {len(decoded_pcm)} bytes, expected "
                f"{expected_pcm_bytes} bytes (tolerance {tolerance})."
            )
        self.assertGreaterEqual(len(decoded_pcm), expected_pcm_bytes)
        self.assertLessEqual(len(decoded_pcm), expected_pcm_bytes + tolerance)
        expected_pcm = decoded_pcm[:expected_pcm_bytes]
        tail_sample_count = 200
        tail_bytes = expected_pcm[-tail_sample_count * 2 :]
        tail_samples = struct.unpack(f"<{tail_sample_count}h", tail_bytes)
        self.assertTrue(all(sample != 0 for sample in tail_samples))


if __name__ == "__main__":
    unittest.main()

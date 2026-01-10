import io
import shutil
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


class TestFFMpegDecoder(unittest.TestCase):
    def test_pcm_passthrough(self) -> None:
        chunks = [b"abc", b"def", b""]
        audio = FakeAudioStream(
            format="pcm",
            sample_rate_hz=16000,
            channels=1,
            chunks=chunks,
        )
        decoded = list(decode_to_pcm_stream(audio))
        self.assertEqual(decoded, chunks)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not available")
    def test_wav_decoding_streams_pcm(self) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b"\x00\x00" * 40)
        wav_bytes = buffer.getvalue()
        audio = FakeAudioStream(
            format="wav",
            sample_rate_hz=8000,
            channels=1,
            chunks=[wav_bytes[:20], wav_bytes[20:]],
        )
        decoder = decode_to_pcm_stream(audio)
        try:
            chunk = next(decoder)
            self.assertTrue(chunk)
        finally:
            decoder.close()

if __name__ == "__main__":
    unittest.main()

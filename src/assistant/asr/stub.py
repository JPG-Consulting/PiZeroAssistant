from __future__ import annotations

from assistant.asr.base import ASRProvider


class StubASRProvider(ASRProvider):
    """
    Stub ASR provider for development and testing.

    Does not perform real transcription.
    """

    def __init__(self, *, return_text: str = "<stub-asr>"):
        self._return_text = return_text

    def transcribe(self, wav_bytes: bytes, sample_rate: int) -> str:
        duration_sec = len(wav_bytes) / (2 * sample_rate)

        print(
            f"[stub-asr] received {len(wav_bytes)} bytes "
            f"({duration_sec:.2f}s @ {sample_rate} Hz)"
        )

        return self._return_text

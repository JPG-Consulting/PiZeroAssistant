from __future__ import annotations

from abc import ABC, abstractmethod


class ASRProvider(ABC):
    """
    Abstract base class for ASR providers.
    """

    @abstractmethod
    def transcribe(self, wav_bytes: bytes, sample_rate: int) -> str:
        """
        Transcribe audio into text.

        Args:
            wav_bytes: WAV PCM bytes (16-bit mono)
            sample_rate: sample rate in Hz (e.g. 16000)

        Returns:
            Transcribed UTF-8 text

        Raises:
            ASRError or subclass on failure
        """
        raise NotImplementedError

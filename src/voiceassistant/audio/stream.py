"""Audio stream abstractions for encoded audio payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Protocol


class AudioStream(Protocol):
    """Streaming encoded audio payload.

    Chunks may be partial or empty, and consumers must not assume frame alignment.
    """

    format: Literal["pcm", "wav", "mp3", "opus"]
    sample_rate_hz: int
    channels: int

    def iter_chunks(self) -> Iterator[bytes]:
        ...

    def close(self) -> None:
        """Optional best-effort cleanup hook for streaming sources."""
        return None


@dataclass(frozen=True)
class BytesAudioStream:
    data: bytes
    format: Literal["pcm", "wav", "mp3", "opus"]
    sample_rate_hz: int
    channels: int
    chunk_size: int = 4096

    def iter_chunks(self) -> Iterator[bytes]:
        for offset in range(0, len(self.data), self.chunk_size):
            yield self.data[offset : offset + self.chunk_size]

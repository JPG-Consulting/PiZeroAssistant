"""Fixed-size ring buffer for byte frames."""

from __future__ import annotations

from typing import List


class ByteRingBuffer:
    """Ring buffer that reuses preallocated bytearrays."""

    def __init__(self, frame_bytes: int, frame_count: int) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        self._frame_bytes = frame_bytes
        self._frames: List[bytearray] = [bytearray(frame_bytes) for _ in range(frame_count)]
        self._index = 0
        self._filled = 0

    def append(self, data: bytes) -> None:
        if len(data) != self._frame_bytes:
            raise ValueError("invalid frame size for ring buffer")
        target = self._frames[self._index]
        target[:] = data
        self._index = (self._index + 1) % len(self._frames)
        self._filled = min(self._filled + 1, len(self._frames))

    def snapshot(self) -> bytes:
        if self._filled == 0:
            return b""
        start = (self._index - self._filled) % len(self._frames)
        ordered = []
        for offset in range(self._filled):
            idx = (start + offset) % len(self._frames)
            ordered.append(bytes(self._frames[idx]))
        return b"".join(ordered)

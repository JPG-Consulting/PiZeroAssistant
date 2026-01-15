"""Incremental Speech Coordinator (ISC)."""

from __future__ import annotations

import enum
import re
import time

from voiceassistant.logging_config import get_logger

logger = get_logger(__name__)


class _State(enum.Enum):
    COLLECTING = "collecting"
    CANCELLED = "cancelled"
    DONE = "done"


_STRONG_PUNCT = {".", "!", "?", "…"}
_CLOSING_PUNCT = {"\"", "'", ")", "]", "}", "»", "”", "’"}
_DANGLING_HYPHENS = {"-", "–", "—"}
_APOSTROPHES = {"'", "’"}

_ABBREVIATIONS = {
    "sr",
    "sra",
    "srta",
    "dr",
    "dra",
    "mr",
    "mrs",
    "ms",
    "mme",
    "mlle",
    "st",
    "prof",
    "etc",
    "vs",
    "e.g",
    "i.e",
    "p.ej",
}

_URL_OR_EMAIL_RE = re.compile(
    r"(https?://\S+|www\.\S+|[^\s@]+@[^\s@]+\.[^\s@]+)$",
    re.IGNORECASE,
)


class IncrementalSpeechCoordinator:
    """Buffers streamed text and emits speakable chunks."""

    def __init__(self) -> None:
        self._state = _State.COLLECTING
        self._buffer = ""
        self._first_delta_time: float | None = None
        self._first_chunk_emitted = False
        self._chunks_emitted = 0

    def push_delta(self, delta: str) -> list[str]:
        """Accepts streamed LLM text and returns zero or more speakable chunks."""
        if self._state is not _State.COLLECTING:
            return []
        if not delta:
            return []
        if self._first_delta_time is None:
            self._first_delta_time = time.monotonic()
        self._buffer += delta
        return self._drain_chunks()

    def finish(self) -> list[str]:
        """Flushes remaining speakable text at end of LLM stream."""
        if self._state is not _State.COLLECTING:
            return []
        self._state = _State.DONE
        if not self._buffer.strip():
            self._buffer = ""
            return []
        chunk = self._buffer
        self._buffer = ""
        if not self._is_stable_boundary(chunk, allow_terminal_alnum=True):
            return []
        self._log_chunk(chunk, "flush_end")
        return [chunk]

    def cancel(self, reason: str) -> None:
        """Cancels the coordinator; no further chunks may be emitted."""
        if self._state is _State.CANCELLED:
            return
        self._state = _State.CANCELLED
        self._buffer = ""
        logger.debug("[ISC] cancelled reason=%s", reason)

    def _drain_chunks(self) -> list[str]:
        chunks: list[str] = []
        while True:
            boundary = self._find_strong_boundary()
            if boundary is not None:
                chunk = self._consume(boundary)
                if chunk:
                    self._log_chunk(chunk, "strong_punct")
                    chunks.append(chunk)
                continue
            boundary = self._find_length_boundary()
            if boundary is not None:
                chunk = self._consume(boundary)
                if chunk:
                    self._log_chunk(chunk, "length_fallback")
                    chunks.append(chunk)
                continue
            break
        return chunks

    def _find_strong_boundary(self) -> int | None:
        for idx in range(len(self._buffer) - 1, -1, -1):
            if self._buffer[idx] not in _STRONG_PUNCT:
                continue
            end = self._extend_boundary(idx)
            candidate = self._buffer[:end]
            if self._is_stable_boundary(candidate, allow_terminal_alnum=False):
                return end
        return None

    def _find_length_boundary(self) -> int | None:
        threshold = 80 if self._chunks_emitted == 0 else 240
        if len(self._buffer) < threshold:
            return None
        whitespace_idx = self._find_last_boundary(lambda ch: ch.isspace(), threshold)
        if whitespace_idx is not None:
            end = self._extend_whitespace(whitespace_idx + 1)
            candidate = self._buffer[:end]
            if self._is_stable_boundary(candidate, allow_terminal_alnum=False):
                return end
        punctuation_idx = self._find_last_boundary(
            lambda ch: ch in {",", ";"},
            threshold,
        )
        if punctuation_idx is not None and punctuation_idx + 1 >= threshold:
            end = self._extend_boundary(punctuation_idx)
            candidate = self._buffer[:end]
            if self._is_stable_boundary(candidate, allow_terminal_alnum=False):
                return end
        return None

    def _find_last_boundary(self, predicate, limit: int) -> int | None:
        start = min(limit, len(self._buffer) - 1)
        for idx in range(start, -1, -1):
            if predicate(self._buffer[idx]):
                return idx
        return None

    def _extend_boundary(self, idx: int) -> int:
        end = idx + 1
        end = self._extend_closing_punct(end)
        end = self._extend_whitespace(end)
        return end

    def _extend_closing_punct(self, start: int) -> int:
        end = start
        while end < len(self._buffer) and self._buffer[end] in _CLOSING_PUNCT:
            end += 1
        return end

    def _extend_whitespace(self, start: int) -> int:
        end = start
        while end < len(self._buffer) and self._buffer[end].isspace():
            end += 1
        return end

    def _consume(self, end: int) -> str:
        chunk = self._buffer[:end]
        self._buffer = self._buffer[end:]
        self._chunks_emitted += 1
        return chunk

    def _is_stable_boundary(self, chunk: str, *, allow_terminal_alnum: bool) -> bool:
        if not chunk:
            return False
        if not chunk.strip():
            return False
        trimmed = chunk.rstrip()
        last_char = trimmed[-1]
        if last_char in _DANGLING_HYPHENS:
            return False
        if last_char in _APOSTROPHES and len(trimmed) >= 2 and trimmed[-2].isalpha():
            return False
        if re.search(r"\d[\.,]$", trimmed):
            return False
        if not chunk.endswith(tuple(" \t\n\r")) and _URL_OR_EMAIL_RE.search(trimmed):
            return False
        if last_char == "." and self._is_abbreviation(trimmed):
            return False
        if last_char.isalnum() and not allow_terminal_alnum and not chunk[-1].isspace():
            return False
        return True

    def _is_abbreviation(self, trimmed: str) -> bool:
        token = self._token_before_period(trimmed)
        if not token:
            return False
        lower = token.lower()
        if lower in {"no", "ok"}:
            return False
        if len(lower) <= 2:
            return True
        return lower in _ABBREVIATIONS

    def _token_before_period(self, trimmed: str) -> str:
        if not trimmed.endswith("."):
            return ""
        idx = len(trimmed) - 2
        letters: list[str] = []
        while idx >= 0 and trimmed[idx].isalpha():
            letters.append(trimmed[idx])
            idx -= 1
        return "".join(reversed(letters))

    def _log_chunk(self, chunk: str, reason: str) -> None:
        if not self._first_chunk_emitted:
            self._first_chunk_emitted = True
            if self._first_delta_time is not None:
                elapsed_ms = (time.monotonic() - self._first_delta_time) * 1000
                logger.debug("[ISC] time_to_first_chunk_ms=%.0f", elapsed_ms)
        logger.debug("[ISC] chunk_len=%d reason=%s", len(chunk), reason)

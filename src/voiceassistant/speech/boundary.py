"""Deterministic sentence boundary evaluation for incremental speech."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

BoundaryConfidence = Literal["strong", "medium", "fallback"]


@dataclass(frozen=True)
class BoundaryDecision:
    kind: Literal["commit", "wait"]
    commit_upto: Optional[int] = None
    confidence: Optional[BoundaryConfidence] = None
    reason: str = ""


class BoundaryEvaluator:
    def __init__(
        self,
        *,
        min_chars_medium: int = 80,
        max_chars_without_boundary: int = 240,
        settle_ms_after_punct: int = 250,
        partial_word_settle_ms: int = 120,
        max_silence_ms: int = 900,
        tail_window: int = 40,
    ) -> None:
        # min_chars_medium: minimum buffered chars before using timing-only boundaries.
        # max_chars_without_boundary: safety valve to avoid unbounded waiting.
        # settle_ms_after_punct: wait time after punctuation before committing.
        # partial_word_settle_ms: short wait before committing around partial word tails.
        # max_silence_ms: silence threshold for timing-based commits.
        # tail_window: only consider boundaries near the end of the buffer.
        self._min_chars_medium = min_chars_medium
        self._max_chars_without_boundary = max_chars_without_boundary
        self._settle_ms_after_punct = settle_ms_after_punct
        self._partial_word_settle_ms = partial_word_settle_ms
        self._max_silence_ms = max_silence_ms
        self._tail_window = tail_window

    def evaluate(
        self,
        *,
        buffer: str,
        last_commit_idx: int,
        now_ms: int,
        last_text_ms: int,
        last_commit_ms: int,
        inside_code_block: bool,
        llm_completed: bool,
    ) -> BoundaryDecision:
        """
        Decide whether a prefix of `buffer` is safe to speak.

        - `buffer` is the full accumulated text
        - `last_commit_idx` is how much has already been spoken
        - All indices refer to positions in `buffer`
        - Must be deterministic and side-effect free
        """
        if last_commit_idx >= len(buffer):
            return BoundaryDecision(kind="wait", reason="no_new_text")

        if inside_code_block:
            return BoundaryDecision(kind="wait", reason="inside_code_block")

        uncommitted = buffer[last_commit_idx:]
        code_spans, has_unclosed = self._code_block_spans(uncommitted)
        if has_unclosed:
            return BoundaryDecision(kind="wait", reason="unclosed_code_block")

        elapsed_since_text = now_ms - last_text_ms
        settle_ok = elapsed_since_text >= self._settle_ms_after_punct
        silence_ok = elapsed_since_text >= self._max_silence_ms
        partial_word_ok = elapsed_since_text >= self._partial_word_settle_ms

        candidate = self._find_punctuation_boundary(
            uncommitted,
            code_spans=code_spans,
        )
        if candidate is not None:
            commit_offset, confidence, reason = candidate
            commit_idx = last_commit_idx + commit_offset
            if commit_idx > last_commit_idx:
                trailing_text = uncommitted[commit_offset:]
                has_trailing_text = bool(trailing_text.strip())
                if settle_ok or llm_completed or has_trailing_text:
                    return BoundaryDecision(
                        kind="commit",
                        commit_upto=commit_idx,
                        confidence=confidence,
                        reason=reason,
                    )
                return BoundaryDecision(kind="wait", reason="punctuation_settle")

        if (
            self._ends_with_partial_word(buffer, llm_completed=llm_completed)
            and partial_word_ok
        ):
            whitespace_offset = self._find_whitespace_boundary(
                uncommitted,
                code_spans=code_spans,
            )
            if whitespace_offset is not None:
                commit_idx = last_commit_idx + whitespace_offset
                if commit_idx > last_commit_idx:
                    return BoundaryDecision(
                        kind="commit",
                        commit_upto=commit_idx,
                        confidence="medium",
                        reason="partial_word_tail",
                    )

        if silence_ok and len(uncommitted) >= self._min_chars_medium:
            whitespace_offset = self._find_whitespace_boundary(
                uncommitted,
                code_spans=code_spans,
            )
            if whitespace_offset is not None:
                commit_idx = last_commit_idx + whitespace_offset
                if commit_idx > last_commit_idx:
                    return BoundaryDecision(
                        kind="commit",
                        commit_upto=commit_idx,
                        confidence="medium",
                        reason="timing_silence",
                    )

        if (
            len(uncommitted) >= self._max_chars_without_boundary
            and silence_ok
        ):
            whitespace_offset = self._find_whitespace_boundary(
                uncommitted,
                code_spans=code_spans,
            )
            if whitespace_offset is not None:
                commit_idx = last_commit_idx + whitespace_offset
                if commit_idx > last_commit_idx:
                    return BoundaryDecision(
                        kind="commit",
                        commit_upto=commit_idx,
                        confidence="fallback",
                        reason="length_safety_valve",
                    )

        if llm_completed:
            commit_idx = len(buffer)
            if commit_idx > last_commit_idx:
                return BoundaryDecision(
                    kind="commit",
                    commit_upto=commit_idx,
                    confidence="fallback",
                    reason="eos_fallback",
                )

        if self._ends_with_partial_word(buffer, llm_completed=llm_completed):
            return BoundaryDecision(kind="wait", reason="partial_word_tail")

        return BoundaryDecision(kind="wait", reason="no_boundary")

    def _ends_with_partial_word(self, buffer: str, *, llm_completed: bool) -> bool:
        if not buffer:
            return False
        if llm_completed:
            return False
        return self._is_word_char(buffer[-1])

    def _code_block_spans(self, text: str) -> tuple[list[tuple[int, int]], bool]:
        spans: list[tuple[int, int]] = []
        idx = 0
        in_block = False
        start = 0
        while True:
            marker = text.find("```", idx)
            if marker == -1:
                break
            if not in_block:
                in_block = True
                start = marker
            else:
                in_block = False
                spans.append((start, marker + 3))
            idx = marker + 3
        if in_block:
            return spans, True
        return spans, False

    def _find_punctuation_boundary(
        self,
        text: str,
        *,
        code_spans: list[tuple[int, int]],
    ) -> Optional[tuple[int, BoundaryConfidence, str]]:
        strong = {"!", "?", "。", "！", "？"}
        medium = {".", ":", ";", "…"}
        tail_start = max(0, len(text) - self._tail_window)
        for idx in range(len(text) - 1, tail_start - 1, -1):
            if self._index_in_spans(idx, code_spans):
                continue
            ch = text[idx]
            if ch not in strong and ch not in medium:
                continue
            next_char = text[idx + 1] if idx + 1 < len(text) else ""
            prev_char = text[idx - 1] if idx - 1 >= 0 else ""
            if ch == "." and self._is_digit(prev_char) and self._is_digit(next_char):
                continue
            if ch in medium and self._is_word_char(next_char):
                continue
            boundary = self._extend_past_whitespace(text, idx + 1)
            if self._splits_word(text, boundary):
                continue
            confidence: BoundaryConfidence = "strong" if ch in strong else "medium"
            return boundary, confidence, "punctuation_boundary"
        return None

    def _find_whitespace_boundary(
        self,
        text: str,
        *,
        code_spans: list[tuple[int, int]],
    ) -> Optional[int]:
        tail_start = max(0, len(text) - self._tail_window)
        for idx in range(len(text) - 1, tail_start - 1, -1):
            if self._index_in_spans(idx, code_spans):
                continue
            if not text[idx].isspace():
                continue
            boundary = self._extend_past_whitespace(text, idx + 1)
            if self._splits_word(text, boundary):
                continue
            return boundary
        return None

    def _extend_past_whitespace(self, text: str, start: int) -> int:
        idx = start
        while idx < len(text) and text[idx].isspace():
            idx += 1
        return idx

    def _splits_word(self, text: str, boundary: int) -> bool:
        if boundary <= 0 or boundary >= len(text):
            return False
        return self._is_word_char(text[boundary - 1]) and self._is_word_char(
            text[boundary]
        )

    def _index_in_spans(self, idx: int, spans: list[tuple[int, int]]) -> bool:
        for start, end in spans:
            if start <= idx < end:
                return True
        return False

    def _is_word_char(self, ch: str) -> bool:
        return ch.isalnum() or ch in {"_", "'", "’"}

    def _is_digit(self, ch: str) -> bool:
        return ch.isdigit()

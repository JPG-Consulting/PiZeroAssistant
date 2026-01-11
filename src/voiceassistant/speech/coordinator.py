"""Incremental speech coordinator scaffolding for buffered text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from voiceassistant.speech.boundary import BoundaryDecision, BoundaryEvaluator


@dataclass(frozen=True)
class SpeakableChunk:
    text: str
    confidence: Optional[str]
    reason: str


class IncrementalSpeechCoordinator:
    """
    Buffers streamed LLM text and emits speakable chunks
    using BoundaryEvaluator.

    This class:
    - does NOT perform TTS
    - does NOT perform playback
    - does NOT spawn threads
    - does NOT block
    """

    def __init__(
        self,
        *,
        evaluator: Optional[BoundaryEvaluator] = None,
        max_pending_chunks: int = 2,
    ) -> None:
        self._evaluator = evaluator or BoundaryEvaluator()
        self._max_pending_chunks = max_pending_chunks
        self.buffer = ""
        self.last_commit_idx = 0
        self.last_text_ms = 0
        self.last_commit_ms = 0
        self.inside_code_block = False
        self.llm_completed = False
        self.cancelled = False
        self.pending_chunks: List[SpeakableChunk] = []

    def on_text(self, fragment: str, *, now_ms: int) -> None:
        """
        Append a fragment of text from the LLM.
        Must not emit chunks directly.
        """
        if self.cancelled:
            return
        if not fragment:
            return
        self.buffer += fragment
        self.last_text_ms = now_ms
        self._update_code_block_state(fragment)
        self._evaluate(now_ms=now_ms)

    def on_llm_complete(self, *, now_ms: int) -> None:
        """
        Signal that no more text will arrive.
        """
        if self.cancelled:
            return
        self.llm_completed = True
        self.last_text_ms = now_ms
        self._evaluate(now_ms=now_ms)

    def cancel(self, *, reason: str) -> None:
        """
        Cancel all pending and future speech.
        Idempotent.
        """
        if self.cancelled:
            return
        self.cancelled = True
        self.pending_chunks.clear()

    def drain_chunks(self) -> List[SpeakableChunk]:
        """
        Return and clear any speakable chunks
        that have been committed so far.
        """
        chunks = list(self.pending_chunks)
        self.pending_chunks.clear()
        return chunks

    def _evaluate(self, *, now_ms: int) -> None:
        if self.cancelled:
            return
        while len(self.pending_chunks) < self._max_pending_chunks:
            decision = self._evaluator.evaluate(
                buffer=self.buffer,
                last_commit_idx=self.last_commit_idx,
                now_ms=now_ms,
                last_text_ms=self.last_text_ms,
                last_commit_ms=self.last_commit_ms,
                inside_code_block=self.inside_code_block,
                llm_completed=self.llm_completed,
            )
            if decision.kind != "commit":
                return
            commit_upto = decision.commit_upto
            if commit_upto is None or commit_upto <= self.last_commit_idx:
                return
            chunk_text = self.buffer[self.last_commit_idx:commit_upto]
            if not chunk_text:
                return
            self.pending_chunks.append(
                SpeakableChunk(
                    text=chunk_text,
                    confidence=decision.confidence,
                    reason=decision.reason,
                )
            )
            self.last_commit_idx = commit_upto
            self.last_commit_ms = now_ms

    def _update_code_block_state(self, fragment: str) -> None:
        if not fragment:
            return
        if fragment.count("```") % 2 == 1:
            self.inside_code_block = not self.inside_code_block

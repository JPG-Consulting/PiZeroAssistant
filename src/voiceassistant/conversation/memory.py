"""Conversation memory storage and persistence."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from voiceassistant.config import ConversationConfig
from voiceassistant.logging_config import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    def __init__(self, config: ConversationConfig) -> None:
        self._config = config
        self._turns: List[Dict[str, object]] = []
        self._last_activity: float | None = None
        self._reset_commands = {
            command.strip().lower() for command in self._config.reset_commands
        }

    @property
    def persistence_enabled(self) -> bool:
        return (
            self._config.enabled
            and self._config.persistence.enabled
            and bool(self._config.persistence.path)
        )

    def add_user(self, text: str) -> None:
        self._add_turn("user", text)

    def add_assistant(self, text: str) -> None:
        self._add_turn("assistant", text)

    def build_messages(self) -> List[Dict[str, str]]:
        return [{"role": turn["role"], "content": turn["text"]} for turn in self._turns]

    def check_and_apply_reset(self, text: str) -> bool:
        normalized = text.strip().lower()
        if normalized in self._reset_commands:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._turns = []
        self._last_activity = None

    def reset_if_idle(self, now: float) -> None:
        if not self._config.enabled:
            return
        if self._last_activity is None:
            return
        if now - self._last_activity >= self._config.reset_after_idle_s:
            self.reset()

    def load_if_enabled(self) -> None:
        if not self.persistence_enabled:
            return
        path = Path(self._config.persistence.path)
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load conversation history")
            return
        if not isinstance(data, list):
            logger.warning("Invalid conversation history format")
            return
        self._turns = []
        for item in data:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            text = item.get("text")
            timestamp = item.get("timestamp")
            if role not in {"user", "assistant"}:
                continue
            if not isinstance(text, str):
                continue
            if not isinstance(timestamp, (int, float)):
                continue
            self._turns.append({"role": role, "text": text, "timestamp": timestamp})
        if self._turns:
            self._last_activity = time.monotonic()
        self._enforce_limits()

    def persist_if_enabled(self) -> None:
        if not self.persistence_enabled:
            return
        path = Path(self._config.persistence.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"role": turn["role"], "text": turn["text"], "timestamp": turn["timestamp"]}
            for turn in self._turns
        ]
        fd, tmp_path = tempfile.mkstemp(prefix="conversation_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _add_turn(self, role: str, text: str) -> None:
        if not self._config.enabled:
            return
        self._turns.append({"role": role, "text": text, "timestamp": time.time()})
        self._last_activity = time.monotonic()
        self._enforce_limits()

    def _enforce_limits(self) -> None:
        if self._config.max_turns:
            max_messages = self._config.max_turns * 2
            if len(self._turns) > max_messages:
                self._turns = self._turns[-max_messages:]
        if self._config.max_chars:
            total_chars = sum(len(turn["text"]) for turn in self._turns)
            while self._turns and total_chars > self._config.max_chars:
                removed = self._turns.pop(0)
                total_chars -= len(removed["text"])

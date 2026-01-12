"""Base definitions for UX LED backends."""

from __future__ import annotations

from typing import Protocol

from voiceassistant.ux.events import UxEvent


class LedUxBackend(Protocol):
    """Backend interface for handling UX events."""

    def handle_event(self, event: UxEvent) -> None:
        """Handle a UX event."""

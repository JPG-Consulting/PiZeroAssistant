"""UX Manager: routes semantic UX events to registered backends."""

from __future__ import annotations

from threading import Thread
from typing import List, Protocol

from voiceassistant.logging_config import get_logger
from voiceassistant.ux.events import UxEvent

logger = get_logger(__name__)


class BackendHandler(Protocol):
    def handle_event(self, event: UxEvent) -> None:
        ...


class UXManager:
    """Best-effort router for UX events."""

    def __init__(self) -> None:
        self._backends: List[BackendHandler] = []

    def register_backend(self, handler: BackendHandler) -> None:
        """Register a backend handler that accepts a UX event."""
        self._backends.append(handler)

    def emit(self, event: UxEvent) -> None:
        """Fan out a UX event to all registered backends.

        Dispatch is non-blocking and never raises to callers.
        """
        if not self._backends:
            return

        for handler in list(self._backends):
            Thread(
                target=self._dispatch,
                args=(handler, event),
                daemon=True,
            ).start()

    def _dispatch(self, handler: BackendHandler, event: UxEvent) -> None:
        try:
            handler.handle_event(event)
        except Exception as exc:  # noqa: BLE001 - must never raise
            logger.warning("UX backend failed for event %s: %s", event, exc)

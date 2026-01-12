"""Semantic, steady-state UX events that form a stable contract.

These events describe what the user should perceive, not transitions or timing.
They are intentionally stable so UX behavior can rely on them as a contract.
"""

from enum import Enum


class UxEvent(str, Enum):
    """Public UX event definitions for the assistant."""

    # Assistant finished startup and is operational; user perceives ready state.
    ASSISTANT_READY = "ASSISTANT_READY"
    # Assistant is idle and listening for wakeword; user sees no LEDs lit.
    ASSISTANT_IDLE = "ASSISTANT_IDLE"
    # Wakeword accepted after cooldown and pre-roll; user perceives wake accepted.
    WAKE_DETECTED = "WAKE_DETECTED"
    # Assistant is actively recording user speech; user perceives listening state.
    LISTENING = "LISTENING"
    # Assistant processing request (STT + LLM); user perceives working state.
    PROCESSING = "PROCESSING"
    # Assistant producing spoken output (TTS playback); user perceives speaking.
    SPEAKING = "SPEAKING"
    # Wakeword detected but no speech recorded; user perceives no-speech feedback.
    NO_SPEECH = "NO_SPEECH"
    # Provider or unexpected pipeline error; user perceives error feedback.
    ERROR = "ERROR"

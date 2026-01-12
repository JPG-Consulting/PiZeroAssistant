"""WAV-based UX audio backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from voiceassistant.audio.playback import PlaybackController, PlaybackRequest
from voiceassistant.logging_config import get_logger
from voiceassistant.ux.audio.base import AudioUxBackend
from voiceassistant.ux.events import UxEvent

logger = get_logger(__name__)

_AUDIO_ROOT = Path(__file__).resolve().parents[2] / "assets" / "ux" / "audio"

_EVENT_AUDIO = {
    UxEvent.ASSISTANT_READY: "ready.wav",
    UxEvent.WAKE_DETECTED: "wake.wav",
    UxEvent.ERROR: "error.wav",
}


class WavAudioBackend(AudioUxBackend):
    """Play UX audio cues from WAV assets."""

    def __init__(self, playback: PlaybackController) -> None:
        self._playback = playback

    def handle_event(self, event: UxEvent) -> None:
        filename = _EVENT_AUDIO.get(event)
        if filename is None:
            return
        if self._playback.is_playing():
            return
        wav_bytes = self._load_wav_bytes(filename)
        if wav_bytes is None:
            return
        try:
            self._playback.play(PlaybackRequest(wav_bytes=wav_bytes))
        except Exception:
            logger.exception("Failed to play UX WAV for event %s", event)

    def _load_wav_bytes(self, filename: str) -> Optional[bytes]:
        asset_path = _AUDIO_ROOT / filename
        if not asset_path.exists():
            logger.warning("Missing UX audio asset: %s", asset_path)
            return None
        try:
            return asset_path.read_bytes()
        except Exception:
            logger.exception("Failed to read UX audio asset: %s", asset_path)
            return None

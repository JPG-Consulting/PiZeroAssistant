"""UX backend bootstrap helpers."""

from __future__ import annotations

from voiceassistant.audio.playback import PlaybackController
from voiceassistant.logging_config import get_logger
from voiceassistant.ux.audio.wav_player import WavAudioBackend
from voiceassistant.ux.leds.respeaker_pihat_apa102 import RespeakerPiHatApa102Backend
from voiceassistant.ux.manager import UXManager

logger = get_logger(__name__)


def setup_ux_backends(manager: UXManager, playback: PlaybackController) -> None:
    """Register UX backends with best-effort error handling."""
    try:
        manager.register_backend(WavAudioBackend(playback))
    except Exception as exc:
        logger.warning("Failed to register WAV UX backend: %s", exc)
    try:
        manager.register_backend(RespeakerPiHatApa102Backend())
    except Exception as exc:
        logger.warning("Failed to register LED UX backend: %s", exc)

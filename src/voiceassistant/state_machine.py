"""Main assistant state machine."""

from __future__ import annotations

import enum
import queue

from voiceassistant.audio.playback import PlaybackController, PlaybackRequest
from voiceassistant.audio.recorder import Recorder, RecordingResult
from voiceassistant.config import AppConfig, resolve_api_key
from voiceassistant.logging_config import get_logger
from voiceassistant.providers.base import ProviderError
from voiceassistant.providers.http import HttpLLMProvider, HttpSTTProvider, HttpTTSProvider
from voiceassistant.providers.router import ProviderRouter, RoutedProvider
from voiceassistant.wakeword.service import WakewordEvent

logger = get_logger(__name__)


class AssistantState(enum.Enum):
    IDLE = "idle"
    WAKE = "wake"
    RECORD = "record"
    STT = "stt"
    LLM = "llm"
    TTS = "tts"


class AssistantStateMachine:
    def __init__(
        self,
        config: AppConfig,
        wakeword_events: queue.Queue[WakewordEvent],
        recorder: Recorder,
        playback: PlaybackController,
    ) -> None:
        self.config = config
        self._wakeword_events = wakeword_events
        self._recorder = recorder
        self._playback = playback
        self._state = AssistantState.IDLE
        self._running = True

        self._stt_router = ProviderRouter(
            providers=[
                RoutedProvider(
                    provider=HttpSTTProvider(
                        name=cfg.name,
                        endpoint=cfg.endpoint,
                        timeout_s=cfg.timeout_s,
                        api_key=resolve_api_key(cfg),
                    ),
                    max_failures=cfg.max_failures,
                    cooldown_s=cfg.cooldown_s,
                )
                for cfg in config.routing.stt_providers
            ],
            max_fallbacks=config.routing.max_fallbacks,
        )
        self._llm_router = ProviderRouter(
            providers=[
                RoutedProvider(
                    provider=HttpLLMProvider(
                        name=cfg.name,
                        endpoint=cfg.endpoint,
                        timeout_s=cfg.timeout_s,
                        api_key=resolve_api_key(cfg),
                    ),
                    max_failures=cfg.max_failures,
                    cooldown_s=cfg.cooldown_s,
                )
                for cfg in config.routing.llm_providers
            ],
            max_fallbacks=config.routing.max_fallbacks,
        )
        self._tts_router = ProviderRouter(
            providers=[
                RoutedProvider(
                    provider=HttpTTSProvider(
                        name=cfg.name,
                        endpoint=cfg.endpoint,
                        timeout_s=cfg.timeout_s,
                        api_key=resolve_api_key(cfg),
                    ),
                    max_failures=cfg.max_failures,
                    cooldown_s=cfg.cooldown_s,
                )
                for cfg in config.routing.tts_providers
            ],
            max_fallbacks=config.routing.max_fallbacks,
        )

    def stop(self) -> None:
        self._running = False

    def run_forever(self) -> None:
        logger.info("Assistant entering IDLE")
        while self._running:
            try:
                event = self._wakeword_events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event:
                self._handle_wake(event)

    def _handle_wake(self, event: WakewordEvent) -> None:
        logger.info("Wake event received")
        self._state = AssistantState.WAKE
        if self.config.wake_beep_path:
            self._play_beep(self.config.wake_beep_path)
        self._state = AssistantState.RECORD
        self._recorder.begin_recording(event.pre_roll_pcm)
        result = self._recorder.wait_for_result(timeout_s=self.config.audio.max_record_seconds + 1)
        if not result:
            logger.warning("Recording timed out")
            self._state = AssistantState.IDLE
            return
        self._recorder.save_wav(result, self.config.record_output_path)

        try:
            transcript = self._run_stt(result)
            reply = self._run_llm(transcript)
            wav_bytes = self._run_tts(reply)
        except ProviderError:
            logger.exception("Provider error in pipeline")
            self._state = AssistantState.IDLE
            return
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected pipeline error")
            self._state = AssistantState.IDLE
            return

        self._state = AssistantState.TTS
        self._playback.play(PlaybackRequest(wav_bytes=wav_bytes))
        self._monitor_playback()
        self._state = AssistantState.IDLE

    def _run_stt(self, result: RecordingResult) -> str:
        self._state = AssistantState.STT
        response, provider_name = self._stt_router.call(
            lambda provider: provider.transcribe(result.pcm)
        )
        logger.info("STT complete via %s", provider_name)
        return response.text

    def _run_llm(self, transcript: str) -> str:
        self._state = AssistantState.LLM
        response, provider_name = self._llm_router.call(lambda provider: provider.complete(transcript))
        logger.info("LLM complete via %s", provider_name)
        return response.text

    def _run_tts(self, text: str) -> bytes:
        self._state = AssistantState.TTS
        response, provider_name = self._tts_router.call(lambda provider: provider.synthesize(text))
        logger.info("TTS complete via %s", provider_name)
        return response.wav_bytes

    def _monitor_playback(self) -> None:
        while self._playback.is_playing():
            try:
                event = self._wakeword_events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event:
                logger.info("Barge-in detected; stopping playback")
                self._playback.stop()
                return

    def _play_beep(self, path: str) -> None:
        try:
            with open(path, "rb") as handle:
                wav_bytes = handle.read()
            self._playback.play(PlaybackRequest(wav_bytes=wav_bytes))
            self._monitor_playback()
        except FileNotFoundError:
            logger.warning("Wake beep not found at %s", path)
        except Exception:
            logger.exception("Failed to play wake beep")

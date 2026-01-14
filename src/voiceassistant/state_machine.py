"""Main assistant state machine."""

from __future__ import annotations

import enum
import io
import queue
import time
import wave
from typing import TYPE_CHECKING

from voiceassistant.audio.playback import PlaybackController, PlaybackRequest
from voiceassistant.audio.recorder import Recorder, RecordingResult
from voiceassistant.conversation.memory import ConversationMemory
from voiceassistant.config import AppConfig
from voiceassistant.llm.prompts import RESET_ACK_TEXT, get_system_prompt
from voiceassistant.logging_config import get_logger
from voiceassistant.providers.base import LLMRequest, ProviderError
from voiceassistant.providers.llm import LLMProvider
from voiceassistant.providers.factory import (
    build_llm_provider,
    build_stt_provider,
    build_tts_provider,
)
from voiceassistant.providers.router import ProviderRouter, RoutedProvider
from voiceassistant.ux.events import UxEvent
from voiceassistant.ux.manager import UXManager
from voiceassistant.wakeword.service import WakewordEvent

logger = get_logger(__name__)

if TYPE_CHECKING:
    from voiceassistant.providers.http import LLMResponse, TTSResponse


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
                    provider=build_stt_provider(cfg),
                    max_failures=cfg.max_failures,
                    cooldown_s=cfg.cooldown_s,
                )
                for cfg in config.routing.stt_providers
            ],
            max_fallbacks=config.routing.max_fallbacks,
        )
        self._llm_limits: dict[str, tuple[Optional[int], bool]] = {}
        llm_providers = []
        for cfg in config.routing.llm_providers:
            provider = build_llm_provider(cfg)
            self._llm_limits[provider.name] = (
                cfg.max_tokens_per_request,
                cfg.provider_type == "lan_http",
            )
            llm_providers.append(
                RoutedProvider(
                    provider=provider,
                    max_failures=cfg.max_failures,
                    cooldown_s=cfg.cooldown_s,
                )
            )
        self._llm_router = ProviderRouter(
            providers=llm_providers,
            max_fallbacks=config.routing.max_fallbacks,
        )
        self._tts_router = ProviderRouter(
            providers=[
                RoutedProvider(
                    provider=build_tts_provider(cfg),
                    max_failures=cfg.max_failures,
                    cooldown_s=cfg.cooldown_s,
                )
                for cfg in config.routing.tts_providers
            ],
            max_fallbacks=config.routing.max_fallbacks,
        )
        self._conversation_memory = ConversationMemory(config.conversation)
        self._conversation_memory.load_if_enabled()
        logger.debug(
            "Conversation persistence enabled: %s",
            self._conversation_memory.persistence_enabled,
        )
        self._ux_manager = UXManager()
        self._emit_ux(UxEvent.ASSISTANT_READY)

    def stop(self) -> None:
        self._running = False

    def run_forever(self) -> None:
        logger.info("Assistant entering IDLE")
        self._emit_ux(UxEvent.ASSISTANT_IDLE)
        while self._running:
            try:
                event = self._wakeword_events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event:
                self._handle_wake(event)

    def _handle_wake(self, event: WakewordEvent) -> None:
        logger.info("Wake event received")
        self._emit_ux(UxEvent.WAKE_DETECTED)
        self._state = AssistantState.WAKE
        if self.config.wake_beep_path:
            self._play_beep(self.config.wake_beep_path)
        self._state = AssistantState.RECORD
        self._recorder.begin_recording(event.pre_roll_pcm)
        self._emit_ux(UxEvent.LISTENING)
        result = self._recorder.wait_for_result(timeout_s=self.config.audio.max_record_seconds + 1)
        if not result:
            logger.warning("Recording timed out")
            self._state = AssistantState.IDLE
            self._emit_ux(UxEvent.ASSISTANT_IDLE)
            return
        self._recorder.save_wav(result, self.config.record_output_path)
        self._emit_ux(UxEvent.PROCESSING)

        try:
            transcript, stt_provider = self._run_stt(result)
            self._conversation_memory.reset_if_idle(time.monotonic())
            if not transcript or not transcript.strip():
                logger.debug(
                    "No speech detected (empty transcript) via %s; returning to IDLE",
                    stt_provider,
                )
                self._state = AssistantState.IDLE
                self._emit_ux(UxEvent.NO_SPEECH)
                self._emit_ux(UxEvent.ASSISTANT_IDLE)
                return
            if self._conversation_memory.check_and_apply_reset(transcript):
                logger.info("Conversation reset via user command")
                self._conversation_memory.persist_if_enabled()
                reply = RESET_ACK_TEXT
            else:
                self._conversation_memory.add_user(transcript)
                messages = self._conversation_memory.build_messages()
                total_chars = sum(len(message["content"]) for message in messages)
                logger.debug(
                    "LLM request: %d messages, %d characters",
                    len(messages),
                    total_chars,
                )
                reply = self._run_llm(messages, get_system_prompt())
                self._conversation_memory.add_assistant(reply)
                self._conversation_memory.persist_if_enabled()
            tts_response, tts_provider = self._run_tts(reply)
            self._log_tts_duration(tts_response)
        except ProviderError:
            logger.exception("Provider error in pipeline")
            self._state = AssistantState.IDLE
            self._emit_ux(UxEvent.ERROR)
            self._emit_ux(UxEvent.ASSISTANT_IDLE)
            return
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected pipeline error")
            self._state = AssistantState.IDLE
            self._emit_ux(UxEvent.ERROR)
            self._emit_ux(UxEvent.ASSISTANT_IDLE)
            return

        self._state = AssistantState.TTS
        self._emit_ux(UxEvent.SPEAKING)
        self._playback.play(
            PlaybackRequest(
                wav_bytes=tts_response.wav_bytes,
                audio=tts_response.audio,
                provider_name=tts_provider,
            )
        )
        self._monitor_playback()
        elapsed = self._playback.get_last_elapsed_time()
        reason = self._playback.get_last_stop_reason()
        if elapsed is None:
            logger.debug(
                "Playback lifecycle: start → stop (reason=%s, elapsed=%.2fs)",
                reason,
                0.0,
            )
        else:
            logger.debug(
                "Playback lifecycle: start → stop (reason=%s, elapsed=%.2fs)",
                reason,
                elapsed,
            )
        self._state = AssistantState.IDLE
        self._emit_ux(UxEvent.ASSISTANT_IDLE)

    def _run_stt(self, result: RecordingResult) -> tuple[str, str]:
        self._state = AssistantState.STT
        wav_bytes = self._build_wav_bytes(result)
        response, provider_name = self._stt_router.call(
            lambda provider: provider.transcribe(wav_bytes)
        )
        logger.info("STT complete via %s", provider_name)
        logger.debug(
            'STT result via %s: "%s" (chars=%d)',
            provider_name,
            response.text,
            len(response.text),
        )
        return response.text, provider_name

    def _run_llm(self, messages: list[dict], system_prompt: str) -> str:
        self._state = AssistantState.LLM
        logger.debug(
            "Using system prompt (present=%s, chars=%d)",
            bool(system_prompt),
            len(system_prompt),
        )

        def _invoke(provider: LLMProvider) -> LLMResponse:
            max_tokens, supports_native = self._llm_limits.get(
                provider.name,
                (None, False),
            )
            # Phase 1 applies limits only via native max_tokens support.
            # Non-native providers ignore limits in Phase 1 to avoid incorrect
            # token estimation/truncation and TTS regressions; Phase 2 handles budgeting.
            request = LLMRequest(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens if supports_native else None,
            )
            return provider.complete(request)

        response, provider_name = self._llm_router.call(_invoke)
        logger.info("LLM complete via %s", provider_name)
        return response.text

    def _run_tts(self, text: str) -> tuple[TTSResponse, str]:
        self._state = AssistantState.TTS
        logger.debug('TTS input text (chars=%d): "%s"', len(text), text)
        response, provider_name = self._tts_router.call(
            lambda provider: provider.synthesize(text)
        )
        logger.info("TTS complete via %s", provider_name)
        return response, provider_name


    def _build_wav_bytes(self, result: RecordingResult) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(result.channels)
            handle.setsampwidth(2)
            handle.setframerate(result.sample_rate_hz)
            handle.writeframes(result.pcm)
        return buffer.getvalue()

    def _log_tts_duration(self, response: TTSResponse) -> None:
        if not response.wav_bytes:
            return
        try:
            with wave.open(io.BytesIO(response.wav_bytes), "rb") as handle:
                frames = handle.getnframes()
                sample_rate_hz = handle.getframerate()
        except wave.Error:
            return
        duration_s = frames / sample_rate_hz
        logger.debug("TTS WAV duration: %.2f seconds", duration_s)

    def _monitor_playback(self) -> None:
        while self._playback.is_playing():
            try:
                event = self._wakeword_events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event:
                logger.debug("Playback barge-in detected")
                logger.info("Barge-in detected; stopping playback")
                self._playback.stop()
                self._playback.set_stop_reason("barge_in")
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

    def _emit_ux(self, event: UxEvent) -> None:
        try:
            self._ux_manager.emit(event)
        except Exception:
            return

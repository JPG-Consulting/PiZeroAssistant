"""Main assistant state machine."""

from __future__ import annotations

import enum
import io
import queue
import time
import wave

from voiceassistant.audio.playback import PlaybackController, PlaybackRequest
from voiceassistant.audio.recorder import Recorder, RecordingResult
from voiceassistant.conversation.memory import ConversationMemory
from voiceassistant.config import AppConfig
from voiceassistant.llm.prompts import RESET_ACK_TEXT, SYSTEM_PROMPT
from voiceassistant.logging_config import get_logger
from voiceassistant.providers.base import LLMRequest, ProviderError
from voiceassistant.providers.http import TTSResponse
from voiceassistant.providers.factory import (
    build_llm_provider,
    build_stt_provider,
    build_tts_provider,
)
from voiceassistant.providers.router import ProviderRouter, RoutedProvider
from voiceassistant.speech.coordinator import IncrementalSpeechCoordinator, SpeakableChunk
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
                    provider=build_stt_provider(cfg),
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
                    provider=build_llm_provider(cfg),
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
            transcript, stt_provider = self._run_stt(result)
            self._conversation_memory.reset_if_idle(time.monotonic())
            if not transcript or not transcript.strip():
                logger.debug(
                    "No speech detected (empty transcript) via %s; returning to IDLE",
                    stt_provider,
                )
                self._state = AssistantState.IDLE
                return
            if self._conversation_memory.check_and_apply_reset(transcript):
                logger.info("Conversation reset via user command")
                self._conversation_memory.persist_if_enabled()
                reply = RESET_ACK_TEXT
            else:
                self._conversation_memory.add_user(transcript)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *self._conversation_memory.build_messages(),
                ]
                total_chars = sum(len(message["content"]) for message in messages)
                logger.debug(
                    "LLM request: %d messages, %d characters",
                    len(messages),
                    total_chars,
                )
                reply = self._try_incremental_speech(messages)
                if reply is None:
                    reply = self._run_llm(messages)
                    self._conversation_memory.add_assistant(reply)
                    self._conversation_memory.persist_if_enabled()
                else:
                    self._conversation_memory.add_assistant(reply)
                    self._conversation_memory.persist_if_enabled()
                    self._state = AssistantState.IDLE
                    return
            tts_response = self._run_tts(reply)
        except ProviderError:
            logger.exception("Provider error in pipeline")
            self._state = AssistantState.IDLE
            return
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected pipeline error")
            self._state = AssistantState.IDLE
            return

        self._state = AssistantState.TTS
        self._playback.play(
            PlaybackRequest(
                wav_bytes=tts_response.wav_bytes,
                audio=tts_response.audio,
            )
        )
        self._monitor_playback()
        self._state = AssistantState.IDLE

    def _run_stt(self, result: RecordingResult) -> tuple[str, str]:
        self._state = AssistantState.STT
        wav_bytes = self._build_wav_bytes(result)
        response, provider_name = self._stt_router.call(
            lambda provider: provider.transcribe(wav_bytes)
        )
        logger.info("STT complete via %s", provider_name)
        return response.text, provider_name

    def _run_llm(self, messages: list[dict]) -> str:
        self._state = AssistantState.LLM
        response, provider_name = self._llm_router.call(
            lambda provider: provider.complete(LLMRequest(messages=messages))
        )
        logger.info("LLM complete via %s", provider_name)
        return response.text

    def _run_tts(self, text: str) -> TTSResponse:
        self._state = AssistantState.TTS
        response, provider_name = self._tts_router.call(lambda provider: provider.synthesize(text))
        logger.info("TTS complete via %s", provider_name)
        return response

    def _build_wav_bytes(self, result: RecordingResult) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(result.channels)
            handle.setsampwidth(2)
            handle.setframerate(result.sample_rate_hz)
            handle.writeframes(result.pcm)
        return buffer.getvalue()

    def _monitor_playback(self) -> bool:
        while self._playback.is_playing():
            try:
                event = self._wakeword_events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event:
                logger.info("Barge-in detected; stopping playback")
                self._playback.stop()
                return True
        return False

    def _try_incremental_speech(self, messages: list[dict]) -> str | None:
        if not self.config.speech.incremental.enabled:
            return None
        provider = self._select_llm_provider()
        if provider is None or not self._llm_supports_streaming(provider):
            return None
        coordinator = IncrementalSpeechCoordinator()
        text_parts: list[str] = []
        reply, spoke_any, had_error = self._stream_llm(
            provider,
            messages,
            coordinator,
            text_parts,
        )
        if had_error and not spoke_any:
            return None
        if not text_parts:
            return None
        return reply

    def _select_llm_provider(self) -> object | None:
        for routed in self._llm_router._eligible():
            return routed.provider
        return None

    def _llm_supports_streaming(self, provider: object) -> bool:
        stream_method = getattr(provider, "stream", None)
        return callable(stream_method)

    def _stream_llm(
        self,
        provider: object,
        messages: list[dict],
        coordinator: IncrementalSpeechCoordinator,
        text_parts: list[str],
    ) -> tuple[str, bool, bool]:
        self._state = AssistantState.LLM
        spoke_any = False
        had_error = False
        try:
            stream_method = getattr(provider, "stream")
            for fragment in stream_method(LLMRequest(messages=messages)):
                if not fragment:
                    continue
                text_parts.append(str(fragment))
                now_ms = self._now_ms()
                coordinator.on_text(str(fragment), now_ms=now_ms)
                chunks = coordinator.drain_chunks()
                if chunks:
                    spoke_any = True
                    if self._play_chunks(chunks):
                        coordinator.cancel(reason="barge_in")
                        break
        except ProviderError:
            had_error = True
        if not coordinator.cancelled:
            now_ms = self._now_ms()
            coordinator.on_llm_complete(now_ms=now_ms)
            chunks = coordinator.drain_chunks()
            if chunks:
                spoke_any = True
                if self._play_chunks(chunks):
                    coordinator.cancel(reason="barge_in")
        return "".join(text_parts), spoke_any, had_error

    def _play_chunks(self, chunks: list[SpeakableChunk]) -> bool:
        for chunk in chunks:
            tts_response = self._run_tts(chunk.text)
            self._playback.play(
                PlaybackRequest(
                    wav_bytes=tts_response.wav_bytes,
                    audio=tts_response.audio,
                )
            )
            if self._monitor_playback():
                return True
        return False

    def _now_ms(self) -> int:
        return int(time.monotonic() * 1000)

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

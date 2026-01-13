import queue
import unittest
from unittest.mock import Mock

from voiceassistant.audio.recorder import RecordingResult
from voiceassistant.config import (
    AppConfig,
    AudioConfig,
    ConversationConfig,
    ConversationPersistenceConfig,
    LoggingConfig,
    ProviderConfig,
    RoutingConfig,
    WakewordConfig,
)
from voiceassistant.state_machine import AssistantState, AssistantStateMachine
from voiceassistant.wakeword.service import WakewordEvent


class FakeRecorder:
    def __init__(self, result: RecordingResult) -> None:
        self._result = result
        self.begin_called = False

    def begin_recording(self, pre_roll_pcm: bytes) -> None:
        self.begin_called = True

    def wait_for_result(self, timeout_s: float) -> RecordingResult:
        return self._result

    def save_wav(self, result: RecordingResult, path: str) -> None:
        return None


class FakePlayback:
    def play(self, request) -> None:
        return None

    def is_playing(self) -> bool:
        return False

    def stop(self) -> None:
        return None


def build_test_config() -> AppConfig:
    stt_provider = ProviderConfig(
        name="stt",
        provider_type="http",
        endpoint="http://localhost:8000/stt",
        timeout_s=5,
        api_key_env=None,
        max_failures=2,
        cooldown_s=60,
        audio_formats=["wav"],
        max_tokens_per_request=None,
        model=None,
        voice=None,
    )
    llm_provider = ProviderConfig(
        name="llm",
        provider_type="local_echo",
        endpoint="",
        timeout_s=5,
        api_key_env=None,
        max_failures=2,
        cooldown_s=60,
        audio_formats=None,
        max_tokens_per_request=None,
        model=None,
        voice=None,
    )
    tts_provider = ProviderConfig(
        name="tts",
        provider_type="http",
        endpoint="http://localhost:8000/tts",
        timeout_s=5,
        api_key_env=None,
        max_failures=2,
        cooldown_s=60,
        audio_formats=None,
        max_tokens_per_request=None,
        model=None,
        voice=None,
    )
    return AppConfig(
        audio=AudioConfig(
            sample_rate_hz=16000,
            channels=1,
            frame_duration_ms=20,
            device=None,
            pre_roll_ms=200,
            max_record_seconds=2,
            record_silence_ms=400,
            vad_mode=2,
        ),
        wakeword=WakewordConfig(
            model_path="dummy.onnx",
            inference_window_ms=80,
            score_threshold=0.6,
            wakeword_cooldown_ms=1500,
        ),
        routing=RoutingConfig(
            stt_providers=[stt_provider],
            llm_providers=[llm_provider],
            tts_providers=[tts_provider],
            max_fallbacks=1,
        ),
        logging=LoggingConfig(
            level="INFO",
            module_levels={},
        ),
        conversation=ConversationConfig(
            enabled=True,
            max_turns=4,
            max_chars=1000,
            reset_after_idle_s=300,
            reset_commands=["reset conversation"],
            persistence=ConversationPersistenceConfig(
                enabled=False,
                path=None,
            ),
        ),
        wake_beep_path=None,
        record_output_path="/tmp/test.wav",
    )


class TestStateMachineEmptyTranscript(unittest.TestCase):
    def test_empty_transcript_short_circuits_pipeline(self) -> None:
        result = RecordingResult(pcm=b"\x00\x00" * 10, sample_rate_hz=16000, channels=1)
        recorder = FakeRecorder(result)
        playback = FakePlayback()
        machine = AssistantStateMachine(
            config=build_test_config(),
            wakeword_events=queue.Queue(),
            recorder=recorder,
            playback=playback,
        )
        machine._run_stt = Mock(return_value=("", "stub-stt"))
        machine._run_llm = Mock()
        machine._run_tts = Mock()

        event = WakewordEvent(timestamp=0.0, pre_roll_pcm=b"")
        machine._handle_wake(event)

        self.assertEqual(machine._state, AssistantState.IDLE)
        machine._run_stt.assert_called_once()
        machine._run_llm.assert_not_called()
        machine._run_tts.assert_not_called()


if __name__ == "__main__":
    unittest.main()

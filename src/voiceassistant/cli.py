"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import queue
import signal
import sys
from dataclasses import dataclass

from voiceassistant.audio.capture import AudioCaptureThread
from voiceassistant.audio.playback import PlaybackController
from voiceassistant.audio.recorder import Recorder
from voiceassistant.config import AppConfig, load_config
from voiceassistant.logging_config import configure_logging, get_logger
from voiceassistant.state_machine import AssistantStateMachine
from voiceassistant.ux.bootstrap import setup_ux_backends
from voiceassistant.ux.manager import UXManager
from voiceassistant.wakeword.service import WakewordEvent, WakewordService

logger = get_logger(__name__)


@dataclass
class AssistantRuntime:
    machine: AssistantStateMachine
    capture: AudioCaptureThread
    wakeword: WakewordService

    def stop(self) -> None:
        self.machine.stop()
        self.wakeword.stop()
        self.capture.stop()


def _build_components(config: AppConfig) -> AssistantRuntime:
    wakeword_queue: queue.Queue = queue.Queue(maxsize=64)
    record_queue: queue.Queue = queue.Queue(maxsize=128)
    event_queue: queue.Queue[WakewordEvent] = queue.Queue(maxsize=8)

    capture = AudioCaptureThread(
        sample_rate_hz=config.audio.sample_rate_hz,
        channels=config.audio.channels,
        frame_duration_ms=config.audio.frame_duration_ms,
        device=config.audio.device,
        wakeword_queue=wakeword_queue,
        record_queue=record_queue,
    )

    wakeword = WakewordService(
        sample_rate_hz=config.audio.sample_rate_hz,
        frame_duration_ms=config.audio.frame_duration_ms,
        inference_window_ms=config.wakeword.inference_window_ms,
        model_path=config.wakeword.model_path,
        score_threshold=config.wakeword.score_threshold,
        wakeword_cooldown_ms=config.wakeword.wakeword_cooldown_ms,
        pre_roll_ms=config.audio.pre_roll_ms,
        vad_mode=config.audio.vad_mode,
        input_queue=wakeword_queue,
        event_queue=event_queue,
    )

    recorder = Recorder(
        frame_duration_ms=config.audio.frame_duration_ms,
        sample_rate_hz=config.audio.sample_rate_hz,
        channels=config.audio.channels,
        vad_mode=config.audio.vad_mode,
        max_record_seconds=config.audio.max_record_seconds,
        record_silence_ms=config.audio.record_silence_ms,
        record_queue=record_queue,
    )

    playback = PlaybackController()

    recorder.start()
    wakeword.start()
    capture.start()

    machine = AssistantStateMachine(
        config=config,
        wakeword_events=event_queue,
        recorder=recorder,
        playback=playback,
    )
    ux_manager: UXManager = machine._ux_manager
    setup_ux_backends(ux_manager, playback)
    return AssistantRuntime(machine=machine, capture=capture, wakeword=wakeword)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Voice Assistant")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(config.logging)

    runtime = _build_components(config)

    def _shutdown(*_args: object) -> None:
        logger.info("Shutdown requested")
        runtime.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        runtime.machine.run_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())

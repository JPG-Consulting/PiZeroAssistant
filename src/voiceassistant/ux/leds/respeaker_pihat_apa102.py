"""ReSpeaker Pi HAT (3x APA102) LED UX backend."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from voiceassistant.logging_config import get_logger
from voiceassistant.ux.events import UxEvent
from voiceassistant.ux.leds.base import LedUxBackend

logger = get_logger(__name__)

OFF = (0, 0, 0)
WHITE = (180, 180, 180)
BLUE = (0, 0, 180)
CYAN = (0, 160, 180)
AMBER = (180, 100, 0)
RED = (180, 0, 0)

FLASH_SECONDS = 0.2
BLINK_ON_SECONDS = 0.15
BLINK_OFF_SECONDS = 0.15
PULSE_CYCLE_SECONDS = 1.8
ROTATE_STEP_SECONDS = 0.25

LED_COUNT = 3


@dataclass(frozen=True)
class _Pattern:
    event: UxEvent


class _Driver:
    """Minimal driver abstraction for APA102 LEDs."""

    def set_pixels(self, colors: Iterable[Tuple[int, int, int]]) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _NoOpDriver(_Driver):
    def set_pixels(self, colors: Iterable[Tuple[int, int, int]]) -> None:
        return None

    def clear(self) -> None:
        return None

    def close(self) -> None:
        return None


class _Apa102Driver(_Driver):
    def __init__(self) -> None:
        try:
            import spidev
        except ImportError as exc:
            raise RuntimeError("spidev not installed") from exc

        self._spi = spidev.SpiDev()
        self._spi.open(0, 0)
        self._spi.mode = 0
        self._spi.max_speed_hz = 8_000_000
        logger.info("APA102 LED driver initialized via spidev (bus=0, device=0)")

    def set_pixels(self, colors: Iterable[Tuple[int, int, int]]) -> None:
        try:
            color_list = list(colors)
            led_colors = [
                color_list[index] if index < len(color_list) else OFF
                for index in range(LED_COUNT)
            ]
            start_frame = [0x00, 0x00, 0x00, 0x00]
            led_frames = []
            for red, green, blue in led_colors:
                led_frames.extend([0xE0 | 0x1F, blue, green, red])
            end_frame = [0xFF] * ((LED_COUNT + 15) // 16)
            self._spi.writebytes(start_frame + led_frames + end_frame)
        except Exception:
            logger.exception("APA102 driver failed while setting pixels")

    def clear(self) -> None:
        try:
            self.set_pixels([OFF] * LED_COUNT)
        except Exception:
            logger.exception("APA102 driver failed while clearing LEDs")

    def close(self) -> None:
        try:
            self.clear()
            self._spi.close()
        except Exception:
            logger.exception("APA102 driver failed while closing")


class RespeakerPiHatApa102Backend(LedUxBackend):
    """Best-effort LED backend for the ReSpeaker Pi HAT."""

    def __init__(self, driver: Optional[_Driver] = None) -> None:
        if driver is not None:
            self._driver = driver
        else:
            try:
                self._driver = _Apa102Driver()
            except Exception:
                logger.warning(
                    "APA102 LED driver unavailable; using no-op backend",
                    exc_info=True,
                )
                self._driver = _NoOpDriver()
        logger.debug(
            "APA102 LED backend running with %s",
            type(self._driver).__name__,
        )
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending_pattern: Optional[_Pattern] = None
        self._stop_event = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="UXLedBackend",
            daemon=True,
        )
        self._thread.start()

    def handle_event(self, event: UxEvent) -> None:
        try:
            with self._condition:
                self._pending_pattern = _Pattern(event=event)
                self._stop_event.set()
                self._condition.notify()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to enqueue LED UX event")

    def close(self) -> None:
        try:
            with self._condition:
                self._closed = True
                self._stop_event.set()
                self._condition.notify()
            self._thread.join(timeout=1.0)
        except Exception:
            logger.exception("Failed to close LED backend")
        finally:
            try:
                self._driver.close()
            except Exception:
                logger.exception("Failed to close LED driver")

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending_pattern is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                pattern = self._pending_pattern
                self._pending_pattern = None
                self._stop_event.clear()
            if pattern is None:
                continue
            try:
                self._apply_pattern(pattern)
            except Exception:
                logger.exception("LED backend failed while rendering pattern")

    def _apply_pattern(self, pattern: _Pattern) -> None:
        event = pattern.event
        if event is UxEvent.ASSISTANT_IDLE:
            self._set_all(OFF)
            return
        if event is UxEvent.ASSISTANT_READY:
            self._flash(WHITE)
            if not self._stop_event.is_set():
                self._set_all(OFF)
            return
        if event is UxEvent.WAKE_DETECTED:
            self._flash(WHITE)
            return
        if event is UxEvent.LISTENING:
            self._set_all(CYAN)
            return
        if event is UxEvent.PROCESSING:
            self._pulse(BLUE)
            return
        if event is UxEvent.SPEAKING:
            self._rotate(WHITE)
            return
        if event is UxEvent.NO_SPEECH:
            self._blink(AMBER)
            if not self._stop_event.is_set():
                self._set_all(OFF)
            return
        if event is UxEvent.ERROR:
            self._pulse(RED)
            return

    def _set_all(self, color: Tuple[int, int, int]) -> None:
        self._driver.set_pixels([color] * LED_COUNT)

    def _flash(self, color: Tuple[int, int, int]) -> None:
        if self._stop_event.is_set():
            return
        self._set_all(color)
        if self._sleep_with_cancel(FLASH_SECONDS):
            return
        self._set_all(OFF)

    def _blink(self, color: Tuple[int, int, int]) -> None:
        for _ in range(2):
            if self._stop_event.is_set():
                return
            self._set_all(color)
            if self._sleep_with_cancel(BLINK_ON_SECONDS):
                return
            self._set_all(OFF)
            if self._sleep_with_cancel(BLINK_OFF_SECONDS):
                return

    def _pulse(self, color: Tuple[int, int, int]) -> None:
        step_count = 18
        step_sleep = PULSE_CYCLE_SECONDS / step_count
        while not self._stop_event.is_set():
            for step in range(step_count):
                if self._stop_event.is_set():
                    return
                factor = self._pulse_factor(step, step_count)
                self._set_all(self._scale_color(color, factor))
                if self._sleep_with_cancel(step_sleep):
                    return

    def _rotate(self, color: Tuple[int, int, int]) -> None:
        index = 0
        while not self._stop_event.is_set():
            colors = [OFF] * LED_COUNT
            colors[index] = color
            self._driver.set_pixels(colors)
            if self._sleep_with_cancel(ROTATE_STEP_SECONDS):
                return
            index = (index + 1) % LED_COUNT

    def _sleep_with_cancel(self, duration: float) -> bool:
        end_time = time.monotonic() + duration
        while time.monotonic() < end_time:
            if self._stop_event.is_set():
                return True
            time.sleep(0.01)
        return False

    @staticmethod
    def _scale_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        return tuple(max(0, min(255, int(channel * factor))) for channel in color)

    @staticmethod
    def _pulse_factor(step: int, step_count: int) -> float:
        half = step_count / 2
        if step < half:
            return step / half
        return (step_count - step) / half

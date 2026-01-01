from __future__ import annotations

from assistant.wakeword.metrics import WakeWordMetrics
from assistant.wakeword.factory import create_wake_detector

import argparse
import logging
import time

from assistant.config import load_config
from assistant.audio import AlsaMicStream

# VAD selection
from assistant.vad import EnergyVAD
try:
    from assistant.vad import WebRtcVAD
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False


def _make_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("assistant")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def main():
    ap = argparse.ArgumentParser(description="Wakeword detector (ALSA + VAD + ONNX)")
    ap.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = _make_logger(cfg.runtime.log_level)

    # Audio stream
    mic = AlsaMicStream(
        sample_rate=cfg.audio.sample_rate,
        block_ms=cfg.audio.block_ms,
        device=cfg.audio.device,
        channels=cfg.audio.channels,
        ring_seconds=max(3.0, cfg.features.clip_seconds + 1.0),
    )

    # VAD
    if cfg.vad.type == "webrtc":
        if not _HAS_WEBRTC:
            raise RuntimeError("vad.type=webrtc but webrtcvad is not available. Install webrtcvad or use energy.")
        vad = WebRtcVAD(
            aggressiveness=cfg.vad.webrtc.aggressiveness,
            sample_rate=cfg.audio.sample_rate,
            frame_ms=cfg.audio.block_ms,
        )
        vad_is_speech = lambda pcm: vad.is_speech(pcm)
        log.info("VAD: WebRTC (aggressiveness=%d)", cfg.vad.webrtc.aggressiveness)
    elif cfg.vad.type == "energy":
        vad = EnergyVAD(
            rms_threshold=cfg.vad.energy.rms_threshold,
            hangover_ms=cfg.vad.energy.hangover_ms,
            block_ms=cfg.audio.block_ms,
            start_frames=getattr(cfg.vad.energy, "start_frames", 1),
        )
        vad_is_speech = lambda pcm: vad.is_speech(pcm, sample_rate=cfg.audio.sample_rate)
        log.info("VAD: Energy (rms_threshold=%.4f)", cfg.vad.energy.rms_threshold)
    else:
        raise ValueError("vad.type must be 'webrtc' or 'energy'")

    # Metrics + detector
    wake_metrics = WakeWordMetrics()
    wake_detector = create_wake_detector(cfg, wake_metrics, mic)

    log.info("Wakeword detector: %s", cfg.wakeword.type)
    log.info("Listening... Ctrl+C to stop.")

    mic.start()
    try:
        last_print = 0.0
        last_report = time.time()
        while True:
            now = time.time()
            if now - last_report > 60:
                wake_metrics.report()
                last_report = now

            mic.poll()  # <-- ALSA read happens here
            fr = mic.pop_frame()
            if fr is None:
                time.sleep(0.001)
                continue

            # VAD on the current block
            speech = vad_is_speech(fr.pcm16)

            # Wake-word detection
            wake_triggered = wake_detector.process_frame(speech)

            if wake_triggered:
                log.info("WAKE triggered")

            # Optional: occasional debug
            if now - last_print > 5.0:
                log.debug("speech=%s", speech)
                last_print = now

    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        mic.stop()
        mic.close()


if __name__ == "__main__":
    main()

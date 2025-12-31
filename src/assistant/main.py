from __future__ import annotations

import argparse
import logging
import time

from assistant.config import load_config
from assistant.audio import AlsaMicStream
from assistant.dsp import LogMelExtractor
from assistant.wakeword import OnnxWakeword

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
        dtype=cfg.audio.dtype,
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
        )
        vad_is_speech = lambda pcm: vad.is_speech(pcm, sample_rate=cfg.audio.sample_rate)
        log.info("VAD: Energy (rms_threshold=%.4f)", cfg.vad.energy.rms_threshold)
    else:
        raise ValueError("vad.type must be 'webrtc' or 'energy'")

    # Feature extractor (must match training)
    feats = LogMelExtractor(
        sr=cfg.audio.sample_rate,
        n_fft=cfg.features.n_fft,
        win_ms=cfg.features.win_ms,
        hop_ms=cfg.features.hop_ms,
        n_mels=cfg.features.n_mels,
        fmin=cfg.features.fmin,
        fmax=cfg.features.fmax,
        log_eps=cfg.features.log_eps,
        clip_seconds=cfg.features.clip_seconds,
    )

    # Wakeword ONNX
    wake = OnnxWakeword(
        model_path=cfg.wakeword.model_path,
        input_name=cfg.wakeword.input_name,
        output_name=cfg.wakeword.output_name,
        threshold=cfg.wakeword.threshold,
        consecutive_hits=cfg.wakeword.consecutive_hits,
        cooldown_ms=cfg.wakeword.cooldown_ms,
    )

    log.info("Wakeword model: %s", cfg.wakeword.model_path)
    log.info("Listening... Ctrl+C to stop.")

    mic.start()
    try:
        last_print = 0.0
        while True:
            mic.poll()  # <-- ALSA read happens here
            fr = mic.pop_frame()
            if fr is None:
                time.sleep(0.001)
                continue


            # VAD on the current block
            speech = vad_is_speech(fr.pcm16)

            # Optional: occasional debug
            now = time.time()
            if now - last_print > 5.0:
                log.debug("speech=%s", speech)
                last_print = now

            if not speech:
                continue

            # When speech is present: score wakeword on the last clip window
            samples = mic.get_last_samples(feats.num_samples)  # int16
            logmel = feats.extract(samples)                    # (1, n_mels, frames)
            p_wake = wake.score(logmel)

            if wake.update(p_wake):
                log.info("WAKEWORD DETECTED (p=%.3f)", p_wake)
            else:
                log.debug("p_wake=%.3f", p_wake)

    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        mic.stop()
        mic.close()


if __name__ == "__main__":
    main()

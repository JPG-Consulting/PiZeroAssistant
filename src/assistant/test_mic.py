from __future__ import annotations

import argparse
import time
import numpy as np

from assistant.config import load_config
from assistant.audio import AlsaMicStream
from assistant.vad import EnergyVAD

# Optional WebRTC VAD
try:
    from assistant.vad import WebRtcVAD
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False


def rms_from_pcm16(pcm16: bytes) -> float:
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def rms_bar(rms: float, width: int = 12) -> str:
    # Typical speech RMS on Pi mics ~0.01–0.05
    scaled = min(rms / 0.05, 1.0)
    filled = int(scaled * width)
    return "█" * filled + "░" * (width - filled)


def main():
    ap = argparse.ArgumentParser(description="Test ALSA microphone and VAD")
    ap.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)

    mic = AlsaMicStream(
        sample_rate=cfg.audio.sample_rate,
        block_ms=cfg.audio.block_ms,
        device=cfg.audio.device,
        channels=cfg.audio.channels,
        ring_seconds=1.0,
    )

    # VAD selection (same logic as main runtime)
    if cfg.vad.type == "webrtc":
        if not _HAS_WEBRTC:
            raise RuntimeError(
                "vad.type=webrtc but webrtcvad is not available. "
                "Install webrtcvad or switch to energy VAD."
            )
        vad = WebRtcVAD(
            aggressiveness=cfg.vad.webrtc.aggressiveness,
            sample_rate=cfg.audio.sample_rate,
            frame_ms=cfg.audio.block_ms,
        )
        vad_is_speech = lambda pcm: vad.is_speech(pcm)
        vad_name = f"WebRTC(aggr={cfg.vad.webrtc.aggressiveness})"
    else:
        vad = EnergyVAD(
            rms_threshold=cfg.vad.energy.rms_threshold,
            hangover_ms=cfg.vad.energy.hangover_ms,
            block_ms=cfg.audio.block_ms,
            start_frames=cfg.vad.energy.start_frames,
            noise_alpha=cfg.vad.energy.noise_alpha,
            noise_factor=cfg.vad.energy.noise_factor,
        )
        vad_is_speech = lambda pcm: vad.is_speech(pcm, cfg.audio.sample_rate)
        vad_name = f"Energy(thr={cfg.vad.energy.rms_threshold:.3f})"

    print("=== test-mic ===")
    print(f"Device      : {cfg.audio.device or 'default'}")
    print(f"Sample rate : {cfg.audio.sample_rate} Hz")
    print(f"Block size  : {cfg.audio.block_ms} ms")
    print(f"VAD         : {vad_name}")
    print("Speak into the microphone. Ctrl+C to stop.\n")

    mic.start()

    try:
        last_print = 0.0
        while True:
            mic.poll()
            fr = mic.pop_frame()
            if fr is None:
                time.sleep(0.001)
                continue

            rms = rms_from_pcm16(fr.pcm16)
            speech = vad_is_speech(fr.pcm16)

            now = time.time()
            if now - last_print >= 0.1:
                bar = rms_bar(rms)
                state = "SPEECH " if speech else "SILENCE"
                print(f"\rRMS: {rms:0.4f} {bar}  VAD: {state}", end="", flush=True)
                last_print = now

    except KeyboardInterrupt:
        print("\n\nStopping test-mic.")
    finally:
        mic.stop()
        mic.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np

from assistant.audio import AlsaMicStream
from assistant.config import load_config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Record wake-word training audio using the ALSA microphone pipeline "
            "(mono int16, 16 kHz). A short fade-in and fade-out is applied to "
            "each recording to avoid audible clicks at clip boundaries."
        )
    )
    ap.add_argument(
        "output",
        type=Path,
        help="Path to the output WAV file",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Recording duration in seconds (default: 3.0)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file",
    )
    return ap.parse_args()


def record(duration_sec: float, output_path: Path, config_path: Path) -> None:
    cfg = load_config(config_path)

    if cfg.audio.dtype not in {"int16", "int16_le", "int16-be", "i2"}:
        raise ValueError(
            "Audio dtype must be 16-bit integer to match runtime pipeline; "
            f"got {cfg.audio.dtype!r}"
        )
    if cfg.audio.channels != 1:
        raise ValueError("Only mono recording is supported")

    target_samples = int(duration_sec * cfg.audio.sample_rate)
    print("=== record-wakeword ===")
    print(f"Device       : {cfg.audio.device or 'default'}")
    print(f"Sample rate  : {cfg.audio.sample_rate} Hz")
    print(f"Block size   : {cfg.audio.block_ms} ms")
    print(f"Channels     : {cfg.audio.channels}")
    print(f"Duration     : {duration_sec:.2f} s")
    print(f"Output file  : {output_path}")

    mic = AlsaMicStream(
        sample_rate=cfg.audio.sample_rate,
        block_ms=cfg.audio.block_ms,
        device=cfg.audio.device,
        channels=cfg.audio.channels,
        ring_seconds=max(1.0, duration_sec),
    )

    frames: list[bytes] = []
    total_samples = 0
    mic.start()
    warmup_end = time.time() + 0.3
    while time.time() < warmup_end:
        mic.poll()
        fr = mic.pop_frame()
        if fr is None:
            time.sleep(0.001)
            continue
    try:
        start = time.time()
        last_print = start
        while total_samples < target_samples:
            mic.poll()
            fr = mic.pop_frame()
            if fr is None:
                time.sleep(0.001)
                continue

            frames.append(fr.pcm16)
            total_samples += len(fr.pcm16) // 2

            now = time.time()
            if now - last_print >= 0.1:
                elapsed = now - start
                done = min(total_samples / target_samples, 1.0)
                print(f"\rRecording: {elapsed:4.1f}s ({done*100:5.1f}%)", end="", flush=True)
                last_print = now
    finally:
        mic.stop()
        mic.close()

    print("\rRecording: done.             ")

    if not frames:
        raise RuntimeError("No audio frames were captured")

    pcm = b"".join(frames)
    needed_bytes = target_samples * 2
    if len(pcm) > needed_bytes:
        pcm = pcm[:needed_bytes]

    samples = np.frombuffer(pcm, dtype=np.int16).copy()
    # Smooth clip boundaries to avoid audible clicks without changing levels.
    samples = apply_fade_i16(samples, fade_ms=8.0, sr=cfg.audio.sample_rate)
    pcm = samples.tobytes()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        with wave.open(f, "wb") as wf:
            wf.setnchannels(cfg.audio.channels)
            wf.setsampwidth(2)
            wf.setframerate(cfg.audio.sample_rate)
            wf.writeframes(pcm)

    print("Saved WAV file.")
    print(
        "Samples    : min={:6d}  max={:6d}  mean={:7.2f}  std={:7.2f}".format(
            int(samples.min()),
            int(samples.max()),
            float(samples.mean()),
            float(samples.std()),
        )
    )


def apply_fade_i16(samples: np.ndarray, fade_ms: float, sr: int) -> np.ndarray:
    """Apply a short linear fade-in and fade-out to int16 samples."""

    if samples.size == 0:
        return samples

    fade_samples = max(1, int(sr * fade_ms / 1000.0))
    fade_len = min(fade_samples, samples.size // 2)
    if fade_len == 0:
        return samples

    faded = samples.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, num=fade_len, endpoint=True, dtype=np.float32)

    faded[:fade_len] *= ramp
    faded[-fade_len:] *= ramp[::-1]

    np.clip(faded, -32768, 32767, out=faded)
    return faded.astype(np.int16)


def main() -> None:
    args = parse_args()
    record(args.duration, args.output, args.config)


if __name__ == "__main__":
    main()

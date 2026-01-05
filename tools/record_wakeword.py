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
            "(mono int16, 16 kHz)."
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

    samples = np.frombuffer(pcm, dtype=np.int16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output_path, "wb") as wf:
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


def main() -> None:
    args = parse_args()
    record(args.duration, args.output, args.config)


if __name__ == "__main__":
    main()

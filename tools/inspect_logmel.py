#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np

from assistant.config import load_config
from assistant.dsp import LogMelExtractor


def read_wav_mono_int16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sr = w.getframerate()
        sampwidth = w.getsampwidth()
        nframes = w.getnframes()
        pcm = w.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(f"sampwidth={sampwidth} (expected 2 for PCM16)")
    if nch != 1:
        raise ValueError(f"channels={nch} (expected mono)")

    return np.frombuffer(pcm, dtype=np.int16), sr


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect log-mel stats for a directory of WAV files")
    ap.add_argument("root", type=Path, help="Directory containing WAV files")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config (default: config/config.yaml)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    feat_cfg = cfg.features
    extractor = LogMelExtractor(
        sr=cfg.audio.sample_rate,
        n_fft=feat_cfg.n_fft,
        win_ms=feat_cfg.win_ms,
        hop_ms=feat_cfg.hop_ms,
        n_mels=feat_cfg.n_mels,
        fmin=feat_cfg.fmin,
        fmax=feat_cfg.fmax,
        log_eps=feat_cfg.log_eps,
        clip_seconds=feat_cfg.clip_seconds,
    )

    wavs = sorted(args.root.rglob("*.wav"))
    if not wavs:
        print(f"No .wav files found under {args.root}")
        return 1

    print(f"Found {len(wavs)} WAV files under {args.root}")
    print(
        "LogMelExtractor: sr={sr} n_fft={n_fft} win_ms={win_ms} hop_ms={hop_ms} "
        "n_mels={n_mels} clip_seconds={clip_seconds}".format(
            sr=extractor.sr,
            n_fft=extractor.n_fft,
            win_ms=extractor.win_ms,
            hop_ms=extractor.hop_ms,
            n_mels=extractor.n_mels,
            clip_seconds=extractor.clip_seconds,
        )
    )

    total_values = 0
    total_sum = 0.0
    total_sumsq = 0.0
    global_min = math.inf
    global_max = -math.inf

    for path in wavs:
        samples, sr = read_wav_mono_int16(path)
        if sr != extractor.sr:
            raise ValueError(f"{path}: sample_rate={sr} (expected {extractor.sr})")

        if len(samples) < extractor.num_samples:
            pad = extractor.num_samples - len(samples)
            samples = np.pad(samples, (0, pad), mode="constant")
        elif len(samples) > extractor.num_samples:
            samples = samples[: extractor.num_samples]

        mel = extractor.extract(samples)
        flat = mel.astype(np.float64).ravel()

        total_values += flat.size
        total_sum += float(np.sum(flat))
        total_sumsq += float(np.sum(flat * flat))
        global_min = min(global_min, float(np.min(flat)))
        global_max = max(global_max, float(np.max(flat)))

    mean = total_sum / total_values
    variance = max(0.0, total_sumsq / total_values - mean * mean)
    std = math.sqrt(variance)

    print("\nLog-mel statistics (all files):")
    print(f"  mean: {mean:.6f}")
    print(f"  std : {std:.6f}")
    print(f"  min : {global_min:.6f}")
    print(f"  max : {global_max:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

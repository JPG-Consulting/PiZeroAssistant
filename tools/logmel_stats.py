#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from assistant.config import load_config
from assistant.dsp import LogMelExtractor


@dataclass
class FileStats:
    path: Path
    mean: float
    std: float
    min: float
    max: float


def read_wav_mono_int16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sr = w.getframerate()
        sampwidth = w.getsampwidth()
        nframes = w.getnframes()
        pcm = w.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(f"{path}: sampwidth={sampwidth} (expected 2 for PCM16)")
    if nch != 1:
        raise ValueError(f"{path}: channels={nch} (expected mono)")

    return np.frombuffer(pcm, dtype=np.int16), sr


def pad_or_trim(samples: np.ndarray, num_samples: int) -> np.ndarray:
    if len(samples) < num_samples:
        pad = num_samples - len(samples)
        return np.pad(samples, (0, pad), mode="constant")
    if len(samples) > num_samples:
        return samples[:num_samples]
    return samples


def compute_file_stats(extractor: LogMelExtractor, path: Path) -> FileStats:
    samples, sr = read_wav_mono_int16(path)
    if sr != extractor.sr:
        raise ValueError(f"{path}: sample_rate={sr} (expected {extractor.sr})")

    samples = pad_or_trim(samples, extractor.num_samples)
    mel = extractor.extract(samples)
    flat = mel.astype(np.float64).ravel()

    return FileStats(
        path=path,
        mean=float(np.mean(flat)),
        std=float(np.std(flat)),
        min=float(np.min(flat)),
        max=float(np.max(flat)),
    )


def aggregate_stats(stats: Iterable[FileStats]) -> tuple[float, float, float, float, float, float]:
    means = []
    stds = []
    global_min = math.inf
    global_max = -math.inf

    for s in stats:
        means.append(s.mean)
        stds.append(s.std)
        global_min = min(global_min, s.min)
        global_max = max(global_max, s.max)

    means_arr = np.array(means, dtype=np.float64)
    stds_arr = np.array(stds, dtype=np.float64)

    mean_of_means = float(np.mean(means_arr))
    std_of_means = float(np.std(means_arr))
    mean_of_stds = float(np.mean(stds_arr))
    std_of_stds = float(np.std(stds_arr))

    return mean_of_means, std_of_means, mean_of_stds, std_of_stds, global_min, global_max


def interpret(stats: list[FileStats], mean_of_means: float, std_of_means: float, std_of_stds: float) -> list[str]:
    messages = []
    if not stats:
        return messages

    # Identify outliers based on per-file mean log-mel values.
    low_threshold = mean_of_means - 3 * std_of_means
    high_threshold = mean_of_means + 3 * std_of_means

    low_energy = [s for s in stats if s.mean < low_threshold]
    high_energy = [s for s in stats if s.mean > high_threshold]

    if low_energy:
        messages.append(
            f"Low-energy files: {len(low_energy)} below mean-3*std ({low_threshold:.4f}); "
            "consider checking microphone gain or background noise levels."
        )
    else:
        messages.append("Low-energy files: none detected relative to dataset mean.")

    if high_energy:
        messages.append(
            f"High-energy files: {len(high_energy)} above mean+3*std ({high_threshold:.4f}); "
            "possible clipping or over-amplification."
        )
    else:
        messages.append("High-energy files: none detected relative to dataset mean.")

    if std_of_means > 1.0 or std_of_stds > 0.5:
        messages.append(
            "Large variance across files: per-file means or stds vary widely; "
            "dataset may mix recording conditions."
        )
    else:
        messages.append("Variance across files appears consistent with expectations.")

    return messages


def write_csv(path: Path, stats: list[FileStats]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "mean", "std", "min", "max"])
        for s in stats:
            writer.writerow([str(s.path), f"{s.mean:.6f}", f"{s.std:.6f}", f"{s.min:.6f}", f"{s.max:.6f}"])


def build_extractor(config_path: Path) -> LogMelExtractor:
    cfg = load_config(config_path)
    feat_cfg = cfg.features
    return LogMelExtractor(
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute dataset log-mel statistics for compatibility checks")
    ap.add_argument("root", type=Path, help="Directory containing WAV files (searched recursively)")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config (default: config/config.yaml)",
    )
    ap.add_argument("--csv", type=Path, help="Optional CSV output path for per-file statistics")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    extractor = build_extractor(args.config)

    wavs = sorted(args.root.rglob("*.wav"))
    if not wavs:
        print(f"No .wav files found under {args.root}")
        return 1

    print(f"Found {len(wavs)} WAV files under {args.root}")
    print(
        "LogMelExtractor: sr={sr} n_fft={n_fft} win_ms={win_ms} hop_ms={hop_ms} n_mels={n_mels} clip_seconds={clip_seconds}".format(
            sr=extractor.sr,
            n_fft=extractor.n_fft,
            win_ms=extractor.win_ms,
            hop_ms=extractor.hop_ms,
            n_mels=extractor.n_mels,
            clip_seconds=extractor.clip_seconds,
        )
    )

    file_stats: list[FileStats] = []
    for path in wavs:
        stats = compute_file_stats(extractor, path)
        file_stats.append(stats)

    (
        mean_of_means,
        std_of_means,
        mean_of_stds,
        std_of_stds,
        global_min,
        global_max,
    ) = aggregate_stats(file_stats)

    print("\nDataset statistics:")
    print(f"  mean of per-file means: {mean_of_means:.6f} (std={std_of_means:.6f})")
    print(f"  mean of per-file stds : {mean_of_stds:.6f} (std={std_of_stds:.6f})")
    print(f"  global min log-mel     : {global_min:.6f}")
    print(f"  global max log-mel     : {global_max:.6f}")

    print("\nInterpretation:")
    for msg in interpret(file_stats, mean_of_means, std_of_means, std_of_stds):
        print(f"- {msg}")

    if args.csv:
        write_csv(args.csv, file_stats)
        print(f"\nWrote per-file statistics to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

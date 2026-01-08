#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def frame_rms(audio: np.ndarray, hop_samples: int) -> np.ndarray:
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")

    total_samples = audio.shape[0]
    if total_samples == 0:
        return np.array([], dtype=np.float32)

    frame_starts = range(0, total_samples, hop_samples)
    rms_values = []
    for start in frame_starts:
        frame = audio[start : start + hop_samples]
        if frame.size == 0:
            continue
        rms_values.append(math.sqrt(float(np.mean(frame ** 2))))

    return np.array(rms_values, dtype=np.float32)


def energy_centroid(rms_values: np.ndarray) -> float | None:
    if rms_values.size == 0:
        return None

    energy_sum = float(np.sum(rms_values))
    if energy_sum == 0.0:
        return None

    indices = np.arange(rms_values.size, dtype=np.float32)
    centroid = float(np.sum(indices * rms_values) / energy_sum)
    return centroid / float(rms_values.size)


def analyze_folder(folder: Path, sample_rate: int, hop_ms: float, verbose: bool) -> None:
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        print(f"[WARN] No WAV files found in {folder}")
        return

    hop_samples = int(round(sample_rate * (hop_ms / 1000.0)))
    if hop_samples <= 0:
        raise ValueError("Hop size too small for sample rate")

    centroids: list[float] = []
    analyzed_files = 0

    for wav in wavs:
        try:
            audio, sr = sf.read(wav, dtype="float32")
        except Exception as exc:
            print(f"[ERROR] {wav.name}: cannot read ({exc})")
            continue

        if sr != sample_rate:
            print(f"[WARN] {wav.name}: sample rate {sr} != expected {sample_rate}, skipping")
            continue

        mono = to_mono(audio)
        rms_values = frame_rms(mono, hop_samples)
        centroid = energy_centroid(rms_values)
        if centroid is None:
            print(f"[WARN] {wav.name}: no energy to analyze, skipping")
            continue

        analyzed_files += 1
        centroids.append(centroid)
        if verbose:
            print(f"{wav.name}\tcentroid={centroid:.4f}")

    if analyzed_files == 0:
        print("[WARN] No files analyzed")
        return

    centroid_array = np.array(centroids, dtype=np.float32)
    mean_position = float(np.mean(centroid_array))
    std_position = float(np.std(centroid_array))
    centered_fraction = float(np.mean((centroid_array >= 0.4) & (centroid_array <= 0.6)))

    print("\nSummary:")
    print(f"  files_analyzed = {analyzed_files}")
    print(f"  mean_position = {mean_position:.4f}")
    print(f"  std = {std_position:.4f}")
    print(f"  centered_fraction = {centered_fraction * 100:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze temporal alignment of wakeword audio by RMS energy centroid"
        )
    )
    ap.add_argument("folder", type=Path, help="Folder containing wakeword WAV files")
    ap.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Expected sample rate (must match runtime/training)",
    )
    ap.add_argument(
        "--hop-ms",
        type=float,
        default=10.0,
        help="Hop size in milliseconds for RMS framing",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file centroid positions",
    )

    args = ap.parse_args()

    analyze_folder(
        folder=args.folder,
        sample_rate=args.sample_rate,
        hop_ms=args.hop_ms,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

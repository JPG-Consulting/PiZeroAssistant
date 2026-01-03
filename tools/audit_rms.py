#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def rms(x: np.ndarray) -> float:
    return math.sqrt(float(np.mean(x ** 2)))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def audit_folder(
    folder: Path,
    rms_min: float,
    rms_max: float,
    peak_max: float,
    move_bad: bool,
):
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        print(f"[WARN] No WAV files found in {folder}")
        return

    bad_dir = folder / "_bad"
    if move_bad:
        bad_dir.mkdir(exist_ok=True)

    stats = {
        "ok": 0,
        "low_rms": 0,
        "high_rms": 0,
        "clipping": 0,
        "total": 0,
    }

    for wav in wavs:
        try:
            audio, sr = sf.read(wav, dtype="float32")
        except Exception as e:
            print(f"[ERROR] {wav.name}: cannot read ({e})")
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        r = rms(audio)
        p = peak(audio)

        stats["total"] += 1

        reason = None
        if r < rms_min:
            stats["low_rms"] += 1
            reason = "LOW_RMS"
        elif r > rms_max:
            stats["high_rms"] += 1
            reason = "HIGH_RMS"

        if p >= peak_max:
            stats["clipping"] += 1
            reason = "CLIPPING"

        if reason:
            print(f"[BAD] {wav.name:30s} rms={r:.4f} peak={p:.3f} -> {reason}")
            if move_bad:
                wav.rename(bad_dir / wav.name)
        else:
            stats["ok"] += 1

    print("\nSummary:")
    for k, v in stats.items():
        print(f"  {k:10s}: {v}")


def main():
    ap = argparse.ArgumentParser(description="Audit RMS / clipping of WAV files")
    ap.add_argument("folder", type=Path, help="Folder containing WAV files")
    ap.add_argument("--rms-min", type=float, default=0.008, help="Minimum RMS")
    ap.add_argument("--rms-max", type=float, default=0.20, help="Maximum RMS")
    ap.add_argument("--peak-max", type=float, default=0.99, help="Peak clipping threshold")
    ap.add_argument(
        "--move-bad",
        action="store_true",
        help="Move bad files into _bad/ subfolder",
    )

    args = ap.parse_args()

    audit_folder(
        folder=args.folder,
        rms_min=args.rms_min,
        rms_max=args.rms_max,
        peak_max=args.peak_max,
        move_bad=args.move_bad,
    )


if __name__ == "__main__":
    main()

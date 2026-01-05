#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from record_wakeword import record


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Record a sequence of wake-word training clips using the existing "
            "ALSA microphone pipeline."
        )
    )
    ap.add_argument(
        "output_dir",
        type=Path,
        help="Directory where recorded WAV files will be stored",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of recordings to capture (default: 10)",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration per recording in seconds (default: 3.0)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== record-dataset ===")
    print(f"Output directory : {args.output_dir}")
    print(f"Recordings       : {args.count}")
    print(f"Duration         : {args.duration:.2f} s")

    for idx in range(1, args.count + 1):
        output_path = args.output_dir / f"recording_{idx:03d}.wav"
        print(f"\n[{idx}/{args.count}] Recording to {output_path}")
        record(args.duration, output_path, args.config)

        if idx != args.count:
            input("Press Enter to start the next recording...")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

def copy_wavs(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for wav in sorted(src.glob("*.wav")):
        shutil.copy2(wav, dst / wav.name)
        count += 1
    return count

def main():
    ap = argparse.ArgumentParser(description="Prepare clean wakeword dataset")
    ap.add_argument("src", type=Path, help="Source dataset root")
    ap.add_argument("dst", type=Path, help="Destination clean dataset root")
    args = ap.parse_args()

    src = args.src
    dst = args.dst

    wake_src = src / "wake"
    not_wake_src = src / "not_wake"

    if not wake_src.is_dir() or not not_wake_src.is_dir():
        raise SystemExit("ERROR: source must contain wake/ and not_wake/")

    wake_dst = dst / "wake"
    not_wake_dst = dst / "not_wake"

    wake_n = copy_wavs(wake_src, wake_dst)
    not_wake_n = copy_wavs(not_wake_src, not_wake_dst)

    print("\nDataset prepared:")
    print(f"  wake     : {wake_n}")
    print(f"  not_wake : {not_wake_n}")
    print(f"  output   : {dst}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import wave
import numpy as np

def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sr = w.getframerate()
        sampwidth = w.getsampwidth()
        nframes = w.getnframes()
        pcm = w.readframes(nframes)
    if sampwidth != 2:
        raise ValueError(f"sampwidth={sampwidth} (expected 2 for PCM16)")
    x = np.frombuffer(pcm, dtype=np.int16)
    if nch == 2:
        # convert stereo to mono by averaging channels
        x = x.reshape(-1, 2).mean(axis=1).astype(np.int16)
        nch = 1
    return x, sr, nch

def rms(x: np.ndarray) -> float:
    xf = x.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(xf * xf) + 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Root folder (e.g., dataset_raw)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-clip-frac", type=float, default=0.01, help="Max fraction clipped samples")
    ap.add_argument("--min-rms", type=float, default=0.005, help="Warn if RMS below this")
    args = ap.parse_args()

    root = Path(args.root)
    wavs = sorted(root.rglob("*.wav"))
    if not wavs:
        print(f"No .wav files found under {root}")
        return 1

    ok = 0
    bad = 0
    for p in wavs:
        try:
            x, sr, nch = read_wav(p)
            dur = len(x) / float(sr)
            r = rms(x)
            clipped = float(np.mean((x == 32767) | (x == -32768)))

            problems = []
            if sr != args.sr:
                problems.append(f"sr={sr} (expected {args.sr})")
            if nch != 1:
                problems.append(f"channels={nch} (expected 1)")
            if dur < args.min_sec:
                problems.append(f"duration={dur:.2f}s (<{args.min_sec}s)")
            if clipped > args.max_clip_frac:
                problems.append(f"clipped={clipped*100:.2f}% (> {args.max_clip_frac*100:.2f}%)")
            if r < args.min_rms:
                problems.append(f"low_rms={r:.4f} (<{args.min_rms})")

            if problems:
                print(f"[WARN] {p}: " + ", ".join(problems))
            else:
                ok += 1

        except Exception as e:
            bad += 1
            print(f"[BAD ] {p}: {e}")

    print(f"Done. ok={ok} bad={bad} total={len(wavs)}")
    return 0 if bad == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())

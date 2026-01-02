#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import wave
import numpy as np
import random

def load_pcm16_mono(path: Path):
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sr = w.getframerate()
        sw = w.getsampwidth()
        n = w.getnframes()
        pcm = w.readframes(n)
    if sw != 2:
        raise ValueError("Expected PCM16 WAV")
    x = np.frombuffer(pcm, dtype=np.int16)
    if nch == 2:
        x = x.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return x, sr

def save_wav(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.astype(np.int16).tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-wake", default="dataset_raw/wake")
    ap.add_argument("--in-neg", default="dataset_raw/negative")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--clip-sec", type=float, default=1.0)
    ap.add_argument("--neg-clips-per-file", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    nclip = int(args.sr * args.clip_sec)

    # Wake clips: centered 1s window
    for p in sorted(Path(args.in_wake).glob("*.wav")):
        x, sr = load_pcm16_mono(p)
        if sr != args.sr or len(x) < nclip:
            continue
        start = max(0, (len(x) - nclip) // 2)
        clip = x[start:start+nclip]
        save_wav(out / "wake" / f"{p.stem}_clip.wav", clip, sr)

    # Negative clips: random windows from long recordings
    for p in sorted(Path(args.in_neg).glob("*.wav")):
        x, sr = load_pcm16_mono(p)
        if sr != args.sr or len(x) < nclip:
            continue
        max_start = len(x) - nclip
        for k in range(args.neg_clips_per_file):
            start = random.randint(0, max_start)
            clip = x[start:start+nclip]
            save_wav(out / "not_wake" / f"{p.stem}_{k:03d}.wav", clip, sr)

    print("Done. Output:", out.resolve())

if __name__ == "__main__":
    main()

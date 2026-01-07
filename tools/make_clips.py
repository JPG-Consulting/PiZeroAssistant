#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import wave
import numpy as np
import random
import hashlib
import logging

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

def stable_file_seed(path: Path, base_seed: int) -> int:
    payload = f"{path.as_posix()}|{base_seed}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-wake", default="dataset_raw/wake")
    ap.add_argument("--in-neg", default="dataset_raw/negative")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--clip-sec", type=float, default=1.0)
    ap.add_argument("--neg-clips-per-file", type=int, default=20)
    ap.add_argument("--neg-clips-per-minute", type=float, default=None)
    ap.add_argument("--neg-hop-sec", type=float, default=None)
    ap.add_argument("--neg-min-rms", type=float, default=None)
    ap.add_argument("--neg-mode", choices=("uniform", "random"), default="random")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.neg_hop_sec is not None and args.neg_hop_sec <= 0:
        raise ValueError("--neg-hop-sec must be positive.")
    if args.neg_hop_sec is not None and args.neg_hop_sec > args.clip_sec:
        logging.warning(
            "--neg-hop-sec (%.2fs) is greater than clip-sec (%.2fs); "
            "this reduces the number of possible negative windows.",
            args.neg_hop_sec,
            args.clip_sec,
        )
    if args.neg_clips_per_minute is not None and args.neg_clips_per_minute < 0:
        raise ValueError("--neg-clips-per-minute must be non-negative.")
    if args.neg_min_rms is not None and args.neg_min_rms < 0:
        raise ValueError("--neg-min-rms must be non-negative.")
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
        neg_hop_sec = args.neg_hop_sec if args.neg_hop_sec is not None else args.clip_sec / 2.0
        hop_samples = max(1, int(args.sr * neg_hop_sec))
        possible_clips = (max_start // hop_samples) + 1
        requested = args.neg_clips_per_file
        density_limit = None
        if args.neg_clips_per_minute is not None:
            duration_minutes = len(x) / sr / 60.0
            density_limit = int(duration_minutes * args.neg_clips_per_minute)
        max_allowed = possible_clips if density_limit is None else min(possible_clips, density_limit)
        effective = min(requested, max_allowed)
        if effective < requested:
            logging.warning(
                "Negative %s: requested %d clips but only %d possible (duration=%.2fs, hop=%.2fs).",
                p.name,
                requested,
                max_allowed,
                len(x) / sr,
                neg_hop_sec,
            )
        if density_limit is not None and density_limit < requested:
            logging.info(
                "Negative %s capped by density to %d clips (limit %.2f clips/min).",
                p.name,
                density_limit,
                args.neg_clips_per_minute,
            )
        if max_start == 0:
            logging.info(
                "Negative %s matches clip length; generating at most one clip.",
                p.name,
            )
        rng = random.Random(stable_file_seed(p, args.seed))
        saved = 0
        skipped_rms = 0
        if args.neg_mode == "uniform":
            for index in range(possible_clips):
                if saved >= effective:
                    break
                start = index * hop_samples
                clip = x[start:start+nclip]
                if args.neg_min_rms is not None:
                    rms = float(np.sqrt(np.mean((clip.astype(np.float32) / 32768.0) ** 2)))
                    if rms < args.neg_min_rms:
                        skipped_rms += 1
                        continue
                save_wav(out / "not_wake" / f"{p.stem}_{saved:03d}.wav", clip, sr)
                saved += 1
        else:
            used_indices: set[int] = set()
            while saved < effective and len(used_indices) < possible_clips:
                index = rng.randrange(possible_clips)
                if index in used_indices:
                    continue
                used_indices.add(index)
                start = index * hop_samples
                clip = x[start:start+nclip]
                if args.neg_min_rms is not None:
                    rms = float(np.sqrt(np.mean((clip.astype(np.float32) / 32768.0) ** 2)))
                    if rms < args.neg_min_rms:
                        skipped_rms += 1
                        continue
                save_wav(out / "not_wake" / f"{p.stem}_{saved:03d}.wav", clip, sr)
                saved += 1
        if saved < effective:
            logging.warning(
                "Negative %s: generated %d/%d clips after RMS filtering.",
                p.name,
                saved,
                effective,
            )
        if skipped_rms:
            logging.info(
                "Negative %s: skipped %d clips below RMS threshold.",
                p.name,
                skipped_rms,
            )

    print("Done. Output:", out.resolve())

if __name__ == "__main__":
    main()

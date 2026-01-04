#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

from assistant.dsp import LogMelExtractor


def main():
    ap = argparse.ArgumentParser(
        description="Gold test: run a WAV file through the wake-word ONNX model"
    )
    ap.add_argument("wav", type=Path, help="Path to WAV file (wake or not-wake)")
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("wakeword.onnx"),
        help="Path to ONNX model (default: wakeword.onnx)",
    )
    args = ap.parse_args()

    assert args.wav.exists(), f"WAV not found: {args.wav}"
    assert args.model.exists(), f"Model not found: {args.model}"

    # -----------------------------
    # Load audio
    # -----------------------------
    wav, sr = sf.read(str(args.wav), dtype="int16")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = wav[:16000]  # force 1s
    if len(wav) < 16000:
        wav = np.pad(wav, (0, 16000 - len(wav)))

    # -----------------------------
    # Feature extraction (MUST match runtime)
    # -----------------------------
    feats = LogMelExtractor(
        sr=16000,
        n_fft=1024,
        win_ms=25,
        hop_ms=10,
        n_mels=40,
        fmin=20,
        fmax=7600,
        log_eps=1e-6,
        clip_seconds=1.0,
    )

    logmel = feats.extract(wav)
    x = logmel[:, None, :, :].astype(np.float32)

    print("logmel mean:", logmel.mean(), "std:", logmel.std())

    # -----------------------------
    # ONNX inference
    # -----------------------------
    sess = ort.InferenceSession(
        str(args.model),
        providers=["CPUExecutionProvider"],
    )

    prob = sess.run(None, {"logmel": x})[0]
    prob = float(prob.squeeze())

    print("prob  :", prob)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

from assistant.config import load_config
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
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config (default: config/config.yaml)",
    )
    args = ap.parse_args()

    assert args.wav.exists(), f"WAV not found: {args.wav}"
    assert args.model.exists(), f"Model not found: {args.model}"

    cfg = load_config(args.config)
    feat_cfg = cfg.features

    # -----------------------------
    # Load audio
    # -----------------------------
    wav, sr = sf.read(str(args.wav), dtype="int16")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != cfg.audio.sample_rate:
        raise ValueError(f"WAV sample_rate={sr} (expected {cfg.audio.sample_rate})")

    num_samples = int(round(cfg.audio.sample_rate * feat_cfg.clip_seconds))
    wav = wav[:num_samples]
    if len(wav) < num_samples:
        wav = np.pad(wav, (0, num_samples - len(wav)))

    # -----------------------------
    # Feature extraction (MUST match runtime)
    # -----------------------------
    feats = LogMelExtractor(
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

    logmel = feats.extract(wav)
    x = logmel[:, None, :, :].astype(np.float32)

    print("input shape:", x.shape)
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

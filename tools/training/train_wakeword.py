"""Minimal wake-word training script for local or Colab use.

Run from repository root so `assistant` imports resolve correctly:

    PYTHONPATH=src python tools/training/train_wakeword.py --data-root data/kws
"""
from __future__ import annotations

import argparse
import math
import random
import json
import subprocess
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Local imports use the runtime feature extractor to ensure contract match
from assistant.dsp import LogMelExtractor

ONNX_INPUT_NAME = "logmel"
ONNX_OUTPUT_NAME = "prob"


@dataclass
class FeatureConfig:
    sample_rate: int
    n_fft: int
    win_ms: int
    hop_ms: int
    n_mels: int
    fmin: float
    fmax: float
    log_eps: float
    clip_seconds: float

    @classmethod
    def from_yaml(cls, path: Path) -> "FeatureConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        feats = cfg["features"]
        audio = cfg["audio"]
        return cls(
            sample_rate=int(audio["sample_rate"]),
            n_fft=int(feats["n_fft"]),
            win_ms=int(feats["win_ms"]),
            hop_ms=int(feats["hop_ms"]),
            n_mels=int(feats["n_mels"]),
            fmin=float(feats["fmin"]),
            fmax=float(feats["fmax"]),
            log_eps=float(feats["log_eps"]),
            clip_seconds=float(feats["clip_seconds"]),
        )


class WakeWordDataset(Dataset):
    def __init__(
        self,
        files: List[Path],
        labels: List[int],
        extractor: LogMelExtractor,
        *,
        augment: bool = False,
        apply_limiter: bool = True,
        debug_logmel_shape: bool = False,
    ):
        self.files = files
        self.labels = labels
        self.extractor = extractor
        self.augment = augment
        self.apply_limiter = apply_limiter
        self.debug_logmel_shape = debug_logmel_shape
        self._checked: set[Path] = set()

    def __len__(self) -> int:
        return len(self.files)

    def _load_audio(self, path: Path) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        target_sr = self.extractor.sr
        if sr != target_sr:
            # High-quality resample
            g = math.gcd(sr, target_sr)
            up = target_sr // g
            down = sr // g
            audio = resample_poly(audio, up, down)
        # Trim/pad to fixed length
        num_samples = self.extractor.num_samples
        original_len = len(audio)
        if len(audio) < num_samples:
            pad = num_samples - len(audio)
            audio = np.pad(audio, (0, pad), mode="constant")
        elif len(audio) > num_samples:
            start = random.randint(0, len(audio) - num_samples)
            audio = audio[start : start + num_samples]

        # Sanity check the raw file levels once to avoid clipping/under-powered data.
        if path not in self._checked:
            rms = float(np.sqrt(np.mean(np.square(audio))))
            peak = float(np.max(np.abs(audio)))
            if original_len < 0.5 * num_samples:
                logging.warning(
                    "File shorter than 50%% of clip_seconds (%.3fs < %.3fs): %s",
                    original_len / self.extractor.sr,
                    num_samples / self.extractor.sr,
                    path,
                )
            if rms < 0.005:
                logging.warning("Low RMS (%.4f) detected for %s; capture levels may be too quiet", rms, path)
            if peak >= 0.999:
                logging.warning("Potential clipping (peak=%.3f) detected for %s; consider lowering record gain", peak, path)
            self._checked.add(path)
        # Optional light augmentation
        if self.augment:
            gain = random.uniform(0.8, 1.2)
            audio = np.clip(audio * gain, -1.0, 1.0)

        # ------------------------------------------------------------------
        # Match the runtime preprocessing path used in OnnxWakeWordDetector.
        # Runtime path (see src/assistant/wakeword/onnx_detector.py):
        #   - ALSA delivers int16 samples captured via record_wakeword.py
        #     or the live stream.
        #   - The detector applies a fixed 2.5x software gain, then a tanh
        #     soft limiter around +/-20000 before feature extraction.
        # We mirror that here by default: convert to int16 PCM, apply the
        # same gain and limiter, and hand the limited int16 to LogMelExtractor.
        # This preserves parity for both record_wakeword.py clips and
        # arecord-style 16-bit PCM WAVs without introducing normalization.
        # ------------------------------------------------------------------
        pcm = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
        pcm_f = pcm.astype(np.float32)
        pcm_f *= 2.5
        if self.apply_limiter:
            pcm_f = np.tanh(pcm_f / 20000.0) * 20000.0
        return pcm_f.astype(np.int16)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wav = self._load_audio(self.files[idx])
        # LogMelExtractor already returns (channels=1, n_mels, frames) to mirror the
        # runtime preprocessing path; avoiding an extra unsqueeze here ensures the
        # DataLoader batches become (B, 1, n_mels, frames), exactly matching the
        # ONNX wake-word input contract.
        logmel = self.extractor.extract(wav)  # (1, n_mels, frames)
        expected_shape = (1, self.extractor.n_mels, self.extractor.num_frames)
        # The shape assertion stays unconditional to catch any parity regressions;
        # --debug-logmel-shape only controls the optional success log below.
        assert (
            logmel.shape == expected_shape
        ), f"Unexpected logmel shape {logmel.shape}; expected {expected_shape}"
        if self.debug_logmel_shape:
            logging.debug("logmel shape ok: %s", logmel.shape)
        logmel = torch.from_numpy(logmel)  # (1, n_mels, frames)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return logmel, label


class SmallCNN(nn.Module):
    """Compact model that adapts to arbitrary feature map shapes."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout(p=0.25)
        # Adaptive pooling keeps the fully connected input size stable as hop/clip lengths change.
        self.adapt_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batching should supply 4D tensors only: (batch, 1, n_mels, frames).
        assert x.dim() == 4 and x.shape[1] == 1, f"Expected 4D input with channel=1, got {tuple(x.shape)}"
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.adapt_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        logit = self.fc(x)
        return logit


def build_dataloaders(
    feature_cfg: FeatureConfig,
    data_root: Path,
    batch_size: int,
    val_split: float,
    seed: int,
    augment: bool,
    apply_limiter: bool,
    debug_logmel_shape: bool,
) -> Tuple[DataLoader, DataLoader, int, int]:
    wake_files = sorted((data_root / "wake").rglob("*.wav"))
    neg_files = sorted((data_root / "not_wake").rglob("*.wav"))
    if not wake_files or not neg_files:
        raise SystemExit("Dataset must contain wake/ and not_wake/ wav files")

    files: List[Path] = wake_files + neg_files
    labels: List[int] = [1] * len(wake_files) + [0] * len(neg_files)

    train_files, val_files, train_labels, val_labels = train_test_split(
        files, labels, test_size=val_split, random_state=seed, stratify=labels
    )

    extractor = LogMelExtractor(
        sr=feature_cfg.sample_rate,
        n_fft=feature_cfg.n_fft,
        win_ms=feature_cfg.win_ms,
        hop_ms=feature_cfg.hop_ms,
        n_mels=feature_cfg.n_mels,
        fmin=feature_cfg.fmin,
        fmax=feature_cfg.fmax,
        log_eps=feature_cfg.log_eps,
        clip_seconds=feature_cfg.clip_seconds,
    )

    train_ds = WakeWordDataset(
        train_files,
        train_labels,
        extractor,
        augment=augment,
        apply_limiter=apply_limiter,
        debug_logmel_shape=debug_logmel_shape,
    )
    val_ds = WakeWordDataset(
        val_files,
        val_labels,
        extractor,
        augment=False,
        apply_limiter=apply_limiter,
        debug_logmel_shape=debug_logmel_shape,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    return train_dl, val_dl, len(wake_files), len(neg_files)


def train_epoch(model, dl, optim, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in tqdm(dl, desc="train", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        optim.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optim.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(1, n)


def evaluate(model, dl, criterion, device, *, desc: str = "val"):
    model.eval()
    total = 0.0
    n = 0
    correct = 0
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc=desc, leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total += loss.item() * len(xb)
            n += len(xb)
            probs = torch.sigmoid(logits)
            correct += ((probs > 0.5) == (yb > 0.5)).sum().item()
            all_probs.append(probs.cpu())
            all_labels.append(yb.cpu())
    if all_probs:
        probs_cat = torch.cat(all_probs).squeeze(1)
        labels_cat = torch.cat(all_labels).squeeze(1)
    else:
        probs_cat = torch.tensor([])
        labels_cat = torch.tensor([])
    return total / max(1, n), correct / max(1, n), probs_cat, labels_cat


def compute_prob_means(probs: torch.Tensor, labels: torch.Tensor) -> Tuple[float | None, float | None]:
    if probs.numel() == 0:
        return None, None
    labels_np = labels.numpy().astype(int)
    probs_np = probs.numpy().astype(float)
    wake_probs = probs_np[labels_np == 1]
    neg_probs = probs_np[labels_np == 0]
    wake_mean = float(np.mean(wake_probs)) if wake_probs.size else None
    neg_mean = float(np.mean(neg_probs)) if neg_probs.size else None
    return wake_mean, neg_mean


def format_prob(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def update_collapse_streak(
    wake_mean: float | None,
    neg_mean: float | None,
    *,
    diff_threshold: float,
    mean_threshold: float,
    current_streak: int,
) -> int:
    if wake_mean is None or neg_mean is None:
        return 0
    mean_diff = abs(wake_mean - neg_mean)
    if mean_diff < diff_threshold and wake_mean > mean_threshold and neg_mean > mean_threshold:
        return current_streak + 1
    return 0


def warn_on_collapse(
    epoch: int,
    wake_mean: float | None,
    neg_mean: float | None,
    *,
    diff_threshold: float,
    mean_threshold: float,
    consecutive_epochs: int,
) -> None:
    if wake_mean is None or neg_mean is None:
        return
    mean_diff = abs(wake_mean - neg_mean)
    if mean_diff >= diff_threshold or wake_mean <= mean_threshold or neg_mean <= mean_threshold:
        return
    logging.warning(
        (
            "Training collapse suspected (epoch %d): wake_mean=%.4f non_wake_mean=%.4f diff=%.4f "
            "for %d consecutive epochs. The model is assigning similarly high probabilities to both "
            "classes, which often means it is not separating wake vs non-wake. Likely causes include "
            "a dataset that is too small or homogeneous, insufficient negative diversity, severe class "
            "imbalance, or overly strong class weighting (pos_weight). Next steps: add diverse negatives, "
            "add near-miss negatives, reduce pos_weight, and inspect dataset balance."
        ),
        epoch,
        wake_mean,
        neg_mean,
        mean_diff,
        consecutive_epochs,
    )


class SigmoidONNXWrapper(nn.Module):
    """Applies sigmoid for export so the runtime sees probabilities."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.base(x))


def export_onnx(model: nn.Module, feature_cfg: FeatureConfig, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    win_length = int(feature_cfg.sample_rate * feature_cfg.win_ms / 1000.0)
    hop_length = int(feature_cfg.sample_rate * feature_cfg.hop_ms / 1000.0)
    num_samples = int(feature_cfg.sample_rate * feature_cfg.clip_seconds)
    num_frames = 1 + (num_samples - win_length) // hop_length
    dummy = torch.zeros(1, 1, feature_cfg.n_mels, num_frames)
    wrapper = SigmoidONNXWrapper(model.eval())
    # Preserve the exact runtime contract used by src/assistant/wakeword/onnx_detector.py:
    # input name "logmel" with shape (1, 1, n_mels, frames) and output name "prob".
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_NAME],
        dynamic_axes=None,  # fixed-shape export (batch=1) keeps IR simple for embedded runtimes
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output}")


def save_metadata(output: Path, feature_cfg: FeatureConfig, args: argparse.Namespace) -> None:
    meta_path = output.with_suffix(".metadata.json")
    try:
        git_hash = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        git_hash = "unknown"

    metadata = {
        "config": str(args.config),
        "feature_config": asdict(feature_cfg),
        "git_hash": git_hash,
        "training_args": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "pos_weight": args.pos_weight,
            "val_split": args.val_split,
            "seed": args.seed,
            "augment": not args.no_augment,
            "limiter": not args.no_limiter,
            "device": args.device,
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a wake-word model compatible with the assistant runtime")
    p.add_argument("--data-root", type=Path, required=True, help="Dataset root containing wake/ and not_wake/ folders")
    p.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Runtime YAML config to mirror features")
    p.add_argument("--output", type=Path, default=Path("models/wakeword.onnx"), help="Destination ONNX path")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Positive class weight for class imbalance (default: auto from dataset ratio)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true", help="Disable random gain augmentation")
    p.add_argument(
        "--no-limiter",
        action="store_true",
        help="Disable the tanh soft limiter (keeps 2.5x gain). ABLATION ONLY — differs from runtime.",
    )
    p.add_argument(
        "--debug-logmel-shape",
        action="store_true",
        help="Assert that extracted log-mel shapes match (1, n_mels, frames) to mirror runtime expectations",
    )
    p.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save training metadata (config + git hash) alongside the exported ONNX",
    )
    p.add_argument(
        "--log-prob-means",
        action="store_true",
        help="Log mean wake/non-wake probabilities each epoch (train + val) for diagnostics",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    apply_limiter = not args.no_limiter
    if apply_limiter:
        logging.info(
            "Using runtime-aligned preprocessing: int16 PCM -> 2.5x gain -> tanh limiter -> LogMelExtractor"
        )
    else:
        logging.warning(
            "Limiter disabled (--no-limiter). This is an ablation path and differs from the runtime OnnxWakeWordDetector."
        )
    if args.debug_logmel_shape:
        logging.info(
            "Debug shape checks enabled: asserting LogMelExtractor outputs (1, n_mels, frames) per sample"
        )

    feature_cfg = FeatureConfig.from_yaml(args.config)
    print("Loaded feature config:", feature_cfg)

    train_dl, val_dl, wake_count, neg_count = build_dataloaders(
        feature_cfg=feature_cfg,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        augment=not args.no_augment,
        apply_limiter=apply_limiter,
        debug_logmel_shape=args.debug_logmel_shape,
    )

    device = torch.device(args.device)
    if wake_count and neg_count:
        ratio = wake_count / neg_count
    else:
        ratio = float("inf")
    balanced = 0.8 <= ratio <= 1.25
    logging.info("Dataset sizes: wake=%d non-wake=%d (ratio=%.2f)", wake_count, neg_count, ratio)
    if args.pos_weight is None:
        if balanced:
            args.pos_weight = 1.0
            logging.info("Balanced dataset detected; using pos_weight=1.0 (no class bias).")
        else:
            args.pos_weight = min(max(neg_count / wake_count, 1.0), 10.0)
            logging.info("Auto pos_weight set to %.2f based on dataset ratio.", args.pos_weight)
            if math.isclose(args.pos_weight, 10.0):
                logging.warning(
                    "pos_weight capped at 10.0; dataset may be severely imbalanced. "
                    "Consider adding more wake samples or diverse negatives."
                )
    elif balanced and not math.isclose(args.pos_weight, 1.0):
        logging.warning(
            "Balanced dataset detected (~1:1) but --pos-weight is %.2f; "
            "this can bias the model toward wake. Consider --pos-weight 1.0.",
            args.pos_weight,
        )
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    collapse_streak = 0
    collapse_diff_threshold = 0.02
    collapse_mean_threshold = 0.8
    collapse_epochs = 3
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc, val_probs, val_labels = evaluate(model, val_dl, criterion, device, desc="val")
        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}%")
        val_wake_mean, val_neg_mean = compute_prob_means(val_probs, val_labels)
        collapse_streak = update_collapse_streak(
            val_wake_mean,
            val_neg_mean,
            diff_threshold=collapse_diff_threshold,
            mean_threshold=collapse_mean_threshold,
            current_streak=collapse_streak,
        )
        if collapse_streak == collapse_epochs:
            warn_on_collapse(
                epoch,
                val_wake_mean,
                val_neg_mean,
                diff_threshold=collapse_diff_threshold,
                mean_threshold=collapse_mean_threshold,
                consecutive_epochs=collapse_epochs,
            )
        if args.log_prob_means:
            _, _, train_probs, train_labels = evaluate(model, train_dl, criterion, device, desc="train")
            train_wake_mean, train_neg_mean = compute_prob_means(train_probs, train_labels)
            print(
                "prob_means train wake={} non_wake={} | val wake={} non_wake={}".format(
                    format_prob(train_wake_mean),
                    format_prob(train_neg_mean),
                    format_prob(val_wake_mean),
                    format_prob(val_neg_mean),
                )
            )
        if val_loss < best_val:
            best_val = val_loss
            export_onnx(model.cpu(), feature_cfg, args.output)
            if args.save_metadata:
                save_metadata(args.output, feature_cfg, args)
            model.to(device)
            print("Saved best model")

    # Inspect validation probabilities for manual threshold selection.
    _, _, probs, labels = evaluate(model, val_dl, criterion, device)
    if probs.numel():
        probs_np = probs.numpy().astype(float)
        labels_np = labels.numpy().astype(int)
        wake_probs = probs_np[labels_np == 1]
        neg_probs = probs_np[labels_np == 0]
        if wake_probs.size:
            print(
                "Wake probs - min: {:.4f}, mean: {:.4f}, max: {:.4f}".format(
                    wake_probs.min(), wake_probs.mean(), wake_probs.max()
                )
            )
        if neg_probs.size:
            print(
                "Non-wake probs - min: {:.4f}, mean: {:.4f}, max: {:.4f}".format(
                    neg_probs.min(), neg_probs.mean(), neg_probs.max()
                )
            )
            suggested = float(np.percentile(neg_probs, 99.9))
            print(
                f"Suggested threshold (99.9th percentile of negatives): {suggested:.4f}. "
                "Tune `wakeword.threshold` around this value in config/config.yaml."
            )

    print("Training complete. Copy the exported ONNX files into models/ for runtime use.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from assistant.audio import AlsaMicStream
from assistant.config import load_config
from assistant.dsp import LogMelExtractor
from assistant.vad import EnergyVAD, WebRtcVAD
from assistant.wakeword.onnx_detector import runtime_preprocess


def build_extractor(cfg) -> LogMelExtractor:
    feat_cfg = cfg.features
    return LogMelExtractor(
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


def rms_normalized(samples_i16: np.ndarray) -> float:
    if samples_i16.size == 0:
        return 0.0
    x = samples_i16.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def infer_window(
    *,
    samples: np.ndarray,
    extractor: LogMelExtractor,
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
) -> tuple[int, float, float, float, float]:
    peak_abs = int(np.max(np.abs(samples))) if samples.size else 0
    rms = rms_normalized(samples)
    logmel = extractor.extract(samples)
    logmel_mean = float(logmel.mean())
    logmel_std = float(logmel.std())

    x = np.ascontiguousarray(logmel, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None, :, :]

    prob = session.run([output_name], {input_name: x})[0]
    prob = float(np.asarray(prob).squeeze())

    return peak_abs, rms, logmel_mean, logmel_std, prob


def print_stats(label: str, stats: tuple[int, float, float, float, float]) -> None:
    peak_abs, rms, logmel_mean, logmel_std, prob = stats
    print(
        f"{label}: peak_abs={peak_abs} rms={rms:.6f} "
        f"logmel_mean={logmel_mean:.6f} logmel_std={logmel_std:.6f} prob={prob:.6f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Capture wakeword inference windows from the ALSA mic path and "
            "evaluate them with the ONNX model."
        )
    )
    ap.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config (default: config/config.yaml)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Path to ONNX model (default: cfg.wakeword.model_path if set, "
            "else models/wakeword.onnx)"
        ),
    )
    ap.add_argument(
        "--outdir",
        default="debug_windows",
        help="Output folder for captured WAVs (default: debug_windows)",
    )
    ap.add_argument(
        "--max-windows",
        type=int,
        default=10,
        help="Stop after capturing N windows (default: 10)",
    )
    ap.add_argument(
        "--cooldown-sec",
        type=float,
        default=1.0,
        help="Cooldown between captures when VAD is active (default: 1.0)",
    )
    ap.add_argument(
        "--no-vad",
        action="store_true",
        help="Capture windows every second regardless of VAD",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = Path(
        args.model
        or cfg.wakeword.model_path
        or "models/wakeword.onnx"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extractor = build_extractor(cfg)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    mic = AlsaMicStream(
        sample_rate=cfg.audio.sample_rate,
        block_ms=cfg.audio.block_ms,
        device=cfg.audio.device,
        channels=cfg.audio.channels,
        ring_seconds=max(3.0, cfg.features.clip_seconds + 1.0),
    )

    vad_is_speech = None
    if not args.no_vad:
        if cfg.vad.type == "webrtc":
            try:
                vad = WebRtcVAD(
                    aggressiveness=cfg.vad.webrtc.aggressiveness,
                    sample_rate=cfg.audio.sample_rate,
                    frame_ms=cfg.audio.block_ms,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "vad.type=webrtc but webrtcvad is not available. "
                    "Install webrtcvad or use energy."
                ) from exc
            vad_is_speech = lambda pcm: vad.is_speech(pcm)
        elif cfg.vad.type == "energy":
            vad = EnergyVAD(
                rms_threshold=cfg.vad.energy.rms_threshold,
                hangover_ms=cfg.vad.energy.hangover_ms,
                block_ms=cfg.audio.block_ms,
                start_frames=cfg.vad.energy.start_frames,
                noise_alpha=cfg.vad.energy.noise_alpha,
                noise_factor=cfg.vad.energy.noise_factor,
            )
            vad_is_speech = lambda pcm: vad.is_speech(pcm, sample_rate=cfg.audio.sample_rate)
        else:
            raise ValueError("vad.type must be 'webrtc' or 'energy'")

    num_samples = int(cfg.audio.sample_rate * cfg.features.clip_seconds)
    windows_captured = 0
    last_capture = 0.0
    next_capture = 0.0

    mic.start()
    try:
        while windows_captured < args.max_windows:
            mic.poll()
            fr = mic.pop_frame()
            if fr is None:
                time.sleep(0.001)
                continue

            now = time.monotonic()
            should_capture = False
            if args.no_vad:
                if now >= next_capture:
                    should_capture = True
                    next_capture = now + 1.0
            else:
                if vad_is_speech is None:
                    raise RuntimeError("VAD requested but not initialized")
                if vad_is_speech(fr.pcm16) and (now - last_capture >= args.cooldown_sec):
                    should_capture = True

            if not should_capture:
                continue

            samples = mic.get_last_samples(num_samples)
            windows_captured += 1
            last_capture = now

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"window_{windows_captured:04d}_{ts}.wav"
            out_path = outdir / filename

            sf.write(str(out_path), samples, cfg.audio.sample_rate, subtype="PCM_16")
            print(f"\nCaptured window: {out_path}")
            print(f"Inference comparison (window {windows_captured}):")

            raw_stats = infer_window(
                samples=samples,
                extractor=extractor,
                session=session,
                input_name=cfg.wakeword.input_name,
                output_name=cfg.wakeword.output_name,
            )
            print_stats("raw", raw_stats)

            processed = runtime_preprocess(samples)
            runtime_stats = infer_window(
                samples=processed,
                extractor=extractor,
                session=session,
                input_name=cfg.wakeword.input_name,
                output_name=cfg.wakeword.output_name,
            )
            print_stats("runtime_preprocess", runtime_stats)

    except KeyboardInterrupt:
        print("\nStopping capture.")
    finally:
        mic.stop()
        mic.close()

    print(f"Captured {windows_captured} window(s).")


if __name__ == "__main__":
    main()

# Wake-word ONNX Pipeline (Current)
This document describes the *contract*, not a step-by-step tutorial.

The runtime code in `src/assistant` is authoritative; this document describes its wake-word build, export, and validation pipeline.

---

## Dataset

- Directory layout (no nested classes):
  ```
  dataset/
  ├── wake/
  └── not_wake/
  ```
- Collect clips with `tools/record_wakeword.py` and `tools/make_clips.py`.
- WAV requirements: mono, 16 kHz, **int16 PCM**. There is no normalization anywhere in the pipeline.

---

## Recording (`tools/record_wakeword.py`)

- Captures raw int16 PCM from ALSA using the runtime device configuration.
- Applies a short fade-in/out to avoid clicks; **no gain, limiter, or normalization** is applied while recording.

---

## Preprocessing parity

### Runtime (`OnnxWakeWordDetector`)
1. Read `int16` PCM from ALSA.
2. Apply fixed 2.5× software gain.
3. Apply tanh soft limiter around ±20000.
4. Cast back to `int16`.
5. Feed into `LogMelExtractor`.

### Training (`train_wakeword.py`)
- Mirrors the runtime path exactly: int16 conversion → 2.5× gain → tanh limiter → `LogMelExtractor`.
- Optional flags: `--no-limiter` (ablation only) and `--debug-logmel-shape`.
- No RMS scaling or normalization at any stage.

---

## Features

- All geometry comes from `config/config.yaml` (sample rate, `n_fft`, `win_ms`, `hop_ms`, `n_mels`, `fmin`, `fmax`, `log_eps`, `clip_seconds`).
- `clip_seconds` may be 1.0 or 2.0 depending on the config; frame counts are derived, not hardcoded.

---

## Model

- Compact CNN (~80k parameters).
- Expects input `(B, 1, n_mels, frames)` and produces logits internally.
- Sigmoid is applied only in the ONNX export wrapper.

---

## ONNX export

- Input name: `logmel`; output name: `prob`.
- Shape: `(1, 1, n_mels, frames)` with **static batch = 1**.
- `opset_version = 18`, `do_constant_folding = True`, no dynamic axes.
- Exported output already includes sigmoid → probability in `[0, 1]`.

---

## Runtime contract

- Runtime consumes the exported probability directly and must **not** apply sigmoid again.

---

## Threshold selection

- Choose thresholds from validation negatives; tune manually in `config/config.yaml`.
- Training does not bake thresholds into the model.

---

## Validation and golden tests

- `tools/gold_test_onnx.py` is the final authority for verifying exported models against the runtime contract.
- Do not rely on Colab-only assumptions; tests must pass locally.

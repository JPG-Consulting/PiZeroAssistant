# Tools

This folder contains small utility scripts for preparing and evaluating wake-word datasets and models. Each entry below lists the key arguments and when they are helpful.

## Audio dataset utilities
- **`record_wakeword.py`** – Records wake-word training audio through the same ALSA microphone path used at runtime so capture conditions match deployment.
  - Arguments: `output` (WAV path), `--duration` (seconds to record, default `3.0`), `--config` (YAML config file, default `config/config.yaml`).
  - When to use: Capture wake-word examples for a training dataset with the exact runtime audio format (mono int16, 16 kHz) and quick level statistics for quality checks.
- **`audit_rms.py`** – Audits a folder of WAV files for RMS/peak levels, optionally moving bad files into a `_bad` subfolder.
  - Arguments: `folder` (path to `.wav` files), `--rms-min`/`--rms-max` (acceptable RMS window, defaults `0.008–0.20`), `--peak-max` (clipping threshold, default `0.99`), `--move-bad` (flag to relocate offenders).
  - When to use: Quickly check incoming recordings for inconsistent levels or clipping before further processing; the thresholds help catch quiet wake words or overdriven negatives.
- **`make_clips.py`** – Generates fixed-length wake and non-wake clips from raw recordings.
  - Arguments: `--in-wake`, `--in-neg` (source folders), `--out` (output dataset root), `--sr` (target sample rate), `--clip-sec` (clip duration), `--neg-clips-per-file` (how many random negatives per long file), `--seed` (repeatable randomness).
  - When to use: Build a balanced, uniform-length dataset from long-form recordings; adjust `clip-sec`/`neg-clips-per-file` to control dataset size and diversity.
- **`prepare_dataset.py`** – Copies cleaned wake and non-wake clips into a new dataset structure.
  - Arguments: `src` (dataset root containing `wake/` and `not_wake/`), `dst` (destination root).
  - When to use: Freeze a curated dataset snapshot after manual review or level auditing.
- **`validate_wavs.py`** – Validates WAV files in a directory tree for sample rate, minimum length, clipping, and RMS thresholds.
  - Arguments: `root` (search root), `--sr` (expected sample rate, default `16000`), `--min-sec` (minimum duration), `--max-clip-frac` (maximum clipped-sample ratio), `--min-rms` (warn if RMS below threshold).
  - When to use: Final sanity-check before training to ensure files meet the model’s audio assumptions.

## Model utilities
- **`gold_test_onnx.py`** – Runs a WAV file through the wake-word ONNX model to inspect the output probability.
  - Arguments: `wav` (1 s clip), `--model` (ONNX path, default `wakeword.onnx`).
  - When to use: Smoke-test a trained model with known wake/non-wake audio and compare probabilities to expectations.
- **`inspect_onnx.py`** – Prints input and output tensor metadata for an ONNX model.
  - Arguments: `model` (ONNX path).
  - When to use: Confirm tensor shapes/dtypes after export, especially when wiring the runtime inference pipeline.
- **`make_dummy_wake_onnx.py`** – Creates a minimal ONNX model that outputs a constant wake probability for testing.
  - Arguments: none; outputs `dummy_wake.onnx`.
  - When to use: Validate the inference path without a trained model; the script pins the ONNX IR version to **7** (compatible with constrained runtimes) and opset to 11 so it runs on older parsers or embedded devices.

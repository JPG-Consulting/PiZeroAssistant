# Tools

This folder contains small utility scripts for preparing and evaluating wake-word datasets and models. Each entry below lists the key arguments and when they are helpful.

> Run these tools from the repository root so configuration files resolve correctly, for example:
>
> ```bash
> PYTHONPATH=src python tools/record_wakeword.py --duration 2.5 data/new_sample.wav
> ```

## Audio dataset utilities
- **`record_wakeword.py`** – Records wake-word training audio through the same ALSA microphone path used at runtime so capture conditions match deployment; applies a short fade-in/out to avoid clicks at clip boundaries.
  - Arguments: `output` (WAV path), `--duration` (seconds to record, default `3.0`), `--config` (YAML config file, default `config/config.yaml`).
  - When to use: Capture wake-word examples for a training dataset with the exact runtime audio format (mono int16, 16 kHz) and quick level statistics for quality checks.
- **`record_dataset.py`** – Collects multiple wake-word recordings in one session using the same pipeline as `record_wakeword.py` without spawning subprocesses.
  - Arguments: `output_dir` (folder for numbered WAV files), `--count` (recording count, default `10`), `--duration` (seconds per clip, default `3.0`), `--prefix` (base filename prefix, default `recording`), `--config` (YAML config file, default `config/config.yaml`).
  - When to use: Capture a batch of wake-word examples interactively with prompts between takes while keeping audio handling identical to the single-recording tool. Use `--prefix` to customize the base filename (e.g., `--prefix wake_01` yields `wake_01_001.wav`).
- **`audit_rms.py`** – Audits a folder of WAV files for RMS/peak levels, optionally moving bad files into a `_bad` subfolder.
  - Arguments: `folder` (path to `.wav` files), `--rms-min`/`--rms-max` (acceptable RMS window, defaults `0.008–0.20`), `--peak-max` (clipping threshold, default `0.99`), `--move-bad` (flag to relocate offenders).
  - When to use: Quickly check incoming recordings for inconsistent levels or clipping before further processing; the thresholds help catch quiet wake words or overdriven negatives.
- **`analyze_wakeword_alignment.py`** – Computes RMS-based energy centroids per frame to measure where wake-word energy falls within each fixed-length clip.
  - Arguments: `folder` (path to `.wav` files), `--sample-rate` (expected sample rate, default `16000`), `--hop-ms` (frame hop in ms, default `10`), `--verbose` (print per-file centroids).
  - When to use: Diagnose temporal alignment bias (e.g., right-aligned wake words) by inspecting energy centroid positions without altering audio or running speech models.
- **`inspect_logmel.py`** – Extracts log-mel features for a directory of WAV files and reports dataset-wide statistics.
  - Arguments: `root` (search root for `.wav` files), `--config` (YAML config file, default `config/config.yaml`).
  - When to use: Compare dataset log-mel distributions against runtime expectations using the same `LogMelExtractor` parameters.
- **`logmel_stats.py`** – Computes per-file and dataset-level log-mel statistics to evaluate compatibility with the trained wake-word model.
  - Arguments: `root` (search root for `.wav` files), `--config` (YAML config file, default `config/config.yaml`), `--csv` (optional CSV export path).
  - When to use: Quantitatively check for low/high-energy outliers and inconsistent variance across recordings; save per-file metrics for deeper review.
  - Notes: The interpretation messages are heuristic and compare files to the rest of the dataset; they highlight concerns but never auto-reject recordings.
  - How to read the results:
    - *Mean of per-file means* shows the typical log-mel level across all clips; clustered means suggest files are recorded at similar loudness.
    - *Std of per-file means* reflects how far those loudness levels spread; a large spread hints that some clips are much quieter or louder than the rest.
    - Worry if the spread is wide and the interpretation mentions many low- or high-energy files; that usually signals inconsistent recording setups rather than a single odd clip.
  - Example: `PYTHONPATH=src python tools/logmel_stats.py data/wake --csv stats.csv`
- **`make_clips.py`** – Generates fixed-length wake and non-wake clips from raw recordings.
  - Arguments: `--in-wake`, `--in-neg` (source folders), `--out` (output dataset root), `--sr` (target sample rate), `--clip-sec` (clip duration), `--neg-clips-per-file` (how many random negatives per long file), `--neg-clips-per-minute` (optional density cap), `--neg-hop-sec` (minimum hop between negative windows, default `clip-sec / 2`), `--neg-min-rms` (optional RMS filter for negatives), `--neg-mode` (`uniform` or `random`), `--seed` (repeatable randomness).
  - When to use: Build a balanced, uniform-length dataset from long-form recordings; adjust `clip-sec`/`neg-clips-per-file`/`neg-clips-per-minute`/`neg-hop-sec`/`neg-min-rms`/`neg-mode` to control dataset size and diversity.
  - Negative clip quality controls:
    - **Negative clip density (`--neg-clips-per-minute`)**: this is a cap, not a target. It limits how many negative clips a file may contribute based on its duration, so short recordings may produce few or zero clips while long recordings can contribute proportionally more. The effective number of clips per file is bounded by `--neg-clips-per-file`, the temporal density (`--neg-clips-per-minute`), and the number of distinct windows possible given `clip-sec` and `neg-hop-sec`. Example: a 20 s recording at 30 clips/min yields up to ~10 clips; a 5 min recording at 30 clips/min yields up to ~150 clips. This avoids duration bias and better reflects real background exposure time.
    - **Minimum RMS filtering for negatives (`--neg-min-rms`)**: filters out extremely low-energy negative clips (near digital silence). RMS is only measured; the audio is never modified. Clips below the threshold are discarded and do not count toward the requested total. This applies **only** to negative clips; wake-word clips are never filtered by RMS. Motivation: silent negatives add little training value and bias the model toward trivial energy-based discrimination, while filtering improves the decision boundary around realistic background noise.
    - **Clipping in negative vs wake-word audio**: clipped negatives are usually low-quality and unrealistic because clipping introduces artificial spectral distortion not representative of typical background noise, so clipping-based rejection (when used) should apply **only** to negatives. Wake-word recordings are **not automatically rejected** due to clipping—loud or clipped wake words can occur in real usage and help robustness; any clipping detection on wake words should be advisory/manual review only.
    - **Design philosophy**: dataset preparation does **not** normalize, rescale, or otherwise modify audio. All decisions are about **which clips are included**, not changing the signal, preserving the original microphone characteristics and runtime conditions.
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
  - When to use: Validate the inference path without a trained model; the script pins the ONNX IR version to **7** (compatible with constrained runtimes) and opset to 11 so it runs on older parsers or embedded devices (legacy / dummy model example). The wake-word runtime contract uses opset 18.
- **`training/`** – Self-contained training helpers for local machines or Google Colab.
  - `train_wakeword.py`: end-to-end trainer that mirrors the runtime feature extractor and exports an ONNX file with the expected `logmel` input and `prob` output. See `tools/training/README.md` for setup and dataset layout.

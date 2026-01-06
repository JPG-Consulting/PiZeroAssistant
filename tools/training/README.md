# Wake-word training (local or Colab)

This folder contains everything needed to train a wake-word model that matches the runtime contract in `src/assistant`. The same script works on a laptop or in Google Colab; the only requirement is a dataset shaped like:

```
<data-root>/
├── wake/       # positive wake-word clips (.wav)
└── not_wake/   # negative/background clips (.wav)
```

## Quick start (local)

1. Install training-only dependencies (kept out of the main runtime requirements):

   ```bash
   pip install -r tools/training/requirements.txt
   ```

2. Run the trainer from the repository root so it can import `assistant` modules:

   ```bash
   PYTHONPATH=src python tools/training/train_wakeword.py \
     --data-root data/kws \
     --output models/wakeword.onnx \
     --epochs 15 \
     --batch-size 64
   ```

3. Copy the exported `wakeword.onnx` (and optional `.onnx.data`) into `models/` for runtime use. The exporter pins the input/output names to `logmel` and `prob` to match `config/config.yaml`.

## Quick start (Google Colab)

1. Bring your dataset into the Colab runtime so it follows the `wake/` and `not_wake/` folder convention:
   - **Upload a ZIP from your computer** (after upload, unpack it):

     ```bash
     # Upload a file named dataset.zip via the Colab file browser, then unpack
     !unzip -q dataset.zip -d data/kws
     ```

   - **Mount from Google Drive** (replace the path if your zip lives elsewhere):

     ```python
     from google.colab import drive
     drive.mount('/content/drive')

     !unzip -q /content/drive/MyDrive/dataset.zip -d data/kws
     ```

2. Clone this repository in a Colab cell and install the training extras:

   ```bash
   !git clone https://github.com/<your-fork>/PiZeroAssistant.git
   %cd PiZeroAssistant
   !pip install -r tools/training/requirements.txt
   ```

3. Run the same training command as above. Colab GPUs are used automatically when available (`--device` defaults to CUDA if present).
4. Download the resulting ONNX files from `models/` to your local machine and drop them into your deployment checkout.

## Script reference

`train_wakeword.py` is intentionally simple so non-experts can iterate quickly:

- Uses the runtime `LogMelExtractor` to guarantee feature parity (16 kHz, 40 mel bins, 1 s window by default). The expected WAV format is 16-bit PCM mono recorded at 16 kHz (e.g., via `tools/record_wakeword.py` which shares the ALSA path with runtime; `arecord`-style PCM WAVs also work).
- Applies the same software gain + tanh soft limiter used by the runtime wake-word detector so training features match deployment behavior.
- Small CNN with ~80k parameters; exports to ONNX with input name `logmel`, output name `prob`, and default opset 11 for compatibility on Raspberry Pi/ONNXRuntime.
- Flags:
  - `--data-root`: required dataset path.
  - `--config`: optional YAML to mirror runtime feature settings (defaults to `config/config.yaml`).
  - `--output`: destination ONNX path (defaults to `models/wakeword.onnx`).
  - `--epochs`, `--batch-size`, `--lr`, `--val-split`, `--pos-weight`, `--seed`, `--device`, `--no-augment`: tune training without editing code.
  - `--no-limiter`: ablation/diagnostic flag that removes the tanh soft limiter (keeps 2.5× gain). Default matches runtime.
  - `--debug-logmel-shape`: assert the extracted log-mel shapes match `(1, n_mels, frames)` for every sample.
  - `--save-metadata`: store a small JSON next to the ONNX export with config and git hash for reproducibility.
- Audio safety checks: the loader logs warnings if clips look clipped or extremely quiet to help avoid distribution shifts.
- After training, the script prints wake/non-wake probability stats and a suggested threshold (99.9th percentile of negatives) to guide manual tuning in `config/config.yaml`.
- Saves the best validation-loss checkpoint directly to ONNX after each epoch.

## Runtime preprocessing parity (analysis)

- **Runtime path (before `LogMelExtractor.extract`)**: `OnnxWakeWordDetector` receives int16 audio from the ALSA mic stream, multiplies by a fixed 2.5× software gain, applies a tanh soft limiter around ±20000, casts back to int16, and only then extracts log-mel features.
- **Recording path**: `tools/record_wakeword.py` uses the same ALSA stream and writes 16-bit PCM mono WAVs (with a short fade at clip boundaries to avoid clicks) without altering gain or applying normalization.
- **Training loader**: `train_wakeword.py` reads those WAVs (or other 16-bit PCM mono 16 kHz files), resamples if needed, pads/trims to `clip_seconds`, and mirrors the runtime preprocessing (int16 conversion → 2.5× gain → tanh limiter) before calling the shared `LogMelExtractor`. Warnings surface if clips are too quiet, clipped, or significantly shorter than the configured clip length.
- **Parity status**: Feature extraction now matches the deployed detector’s preprocessing; keeping `config/config.yaml` consistent between training and runtime ensures the ONNX model sees the same frame/mel geometry as production.

## Tips for better results

- Use `tools/record_wakeword.py` and `tools/make_clips.py` to gather consistent, balanced clips.
- Keep `clip_seconds` in your dataset aligned with `config/config.yaml` (1.0 s by default). The script pads/trims automatically but matching lengths reduces artifacts.
- Start with a generous negative set (background speech/noise) to reduce false positives, then fine-tune the `wakeword.threshold` in `config/config.yaml` after deployment.

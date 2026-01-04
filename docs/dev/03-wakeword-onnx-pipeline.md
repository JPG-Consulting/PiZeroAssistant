# Wake-word ONNX Pipeline (Final)

This document describes the **canonical, end-to-end pipeline** used to build,
validate, train, export, and deploy an **offline wake-word detector** using ONNX.

It is designed to be:
- Reproducible
- Hardware-agnostic
- Aligned with the current Google Colab training notebook

No local machine–specific paths are assumed.

---

## 1. Dataset Structure

The dataset **must** follow this structure:

```
dataset/
├── wake/
│   ├── wake_001.wav
│   ├── wake_002.wav
│   └── ...
└── not_wake/
    ├── noise_001.wav
    ├── speech_001.wav
    └── ...
```

All WAV files must be:
- Mono
- 16 kHz sample rate
- PCM (`int16` or `float32`)

---

## 2. WAV Validation

Before any training, validate WAV integrity:

```bash
python tools/validate_wavs.py path/to/dataset/wake
python tools/validate_wavs.py path/to/dataset/not_wake
```

This checks:
- WAV readability
- Sample rate
- Channel count

No files should fail this step.

---

## 3. RMS & Clipping Audit

Next, audit signal levels:

```bash
python tools/audit_rms.py path/to/dataset/wake
python tools/audit_rms.py path/to/dataset/not_wake
```

Files may be flagged as:
- `LOW_RMS` (too quiet)
- `HIGH_RMS`
- `CLIPPING`

Flagged files are automatically moved into a `_bad/` subfolder and **excluded**
from training.

---

## 4. Preparing the Training ZIP

Create a ZIP archive containing the cleaned dataset.

**Important rules**
- The folder *inside the ZIP* must be named exactly `dataset/`
- The ZIP filename itself is arbitrary

Example:

```bash
zip -r dataset.zip dataset
```

---

## 5. Google Colab Environment Setup

### 5.1 Install dependencies

```python
!pip -q install numpy==2.* torch torchvision torchaudio torchcodec onnx onnxruntime onnxscript tqdm
```

`torchcodec` is required and **must** be installed.

### 5.2 Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

---

## 6. Dataset Extraction in Colab

```python
import zipfile
from pathlib import Path

ZIP_PATH = Path("/content/drive/MyDrive/wakeword-training/dataset.zip")
OUT_DIR  = Path("/content/dataset")

assert ZIP_PATH.exists()
OUT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(OUT_DIR)
```

The notebook **auto-detects** whether the dataset is located at:
- `/content/dataset`
- `/content/dataset/dataset`

---

## 7. Canonical Feature Parameters (DO NOT CHANGE)

These parameters **must match runtime**:

- Sample rate: 16 kHz
- Log-Mel features
- 40 mel bins
- 25 ms window
- 10 ms hop
- 1.0 s context
- 98 frames

Any mismatch will break inference.

---

## 8. Model Architecture

The model is a **TinyDSCNN-style binary classifier**:

- Depthwise-separable convolutions
- Global convolution over `(40 × 98)`
- Single logit output

The model outputs **logits**, not probabilities.

---

## 9. Training

Training uses:

- `BCEWithLogitsLoss`
- Strong class imbalance correction via `pos_weight`
- Adam optimizer
- Fixed-shape batches

Validation loss is monitored; training can be stopped early if it plateaus.

---

## 10. Threshold Selection

After training, validation probabilities are analyzed:

- Wake probabilities
- Not-wake probabilities

A **conservative threshold** is chosen:

```
threshold = max(neg_probs) × 1.2
```

This ensures near-zero false positives.

Threshold selection is **explicitly separated** from training.

---

## 11. ONNX Export

The model is exported with:

- Static input shape `(1, 1, 40, 98)`
- `opset_version = 11` (ARM-safe)
- Output name: `prob`

The ONNX model outputs a **probability**, not logits.

Both files must be preserved:
- `wakeword.onnx`
- `wakeword.onnx.data`

---

## 12. Model Storage

Exported models are stored in Google Drive:

```
MyDrive/
└── wakeword-training/
    ├── wakeword.onnx
    └── wakeword.onnx.data
```

Only the ONNX artifacts are committed or deployed — **never training checkpoints**.

---

## 13. Runtime Contract

### Input

```
logmel: float32 (1, 1, 40, 98)
```

### Output

```
prob: float32 (1, 1)
```

The runtime **must not apply sigmoid** again.

---

## 14. Golden Test Script

A canonical script (`tools/gold_test_onnx.py`) is provided to validate the ONNX
model on real WAV files **outside Colab**.

This script is the final authority for model correctness.

---

## 15. Final Notes

- The repository defines the **interface**, not the trained weights
- Training and deployment are strictly separated
- Any future model must conform to this contract

This document is the **single source of truth** for the wake-word pipeline.

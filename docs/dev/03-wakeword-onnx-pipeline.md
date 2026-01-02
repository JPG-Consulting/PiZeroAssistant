# Wake-word ONNX Pipeline (Developer Notes)

## Overview
This project uses a wake-word detector running fully offline on Raspberry Pi Zero 2 W.
Wake-word models are trained externally and deployed as ONNX files.

This document describes:
- the ONNX input/output contract
- the role of the dummy wake-word model
- how inference is wired in the assistant
- how training artifacts are brought into Google Colab

---

## Dummy Wake-word Model (Smoke Test)

A dummy ONNX model (`dummy_wake.onnx`) is used during development to validate:
- ONNXRuntime availability on ARM
- feature extraction correctness
- detector cooldown and metrics logic

This model is **not** a real wake-word detector.
It always outputs a fixed wake probability.

### Why it exists
- isolate infrastructure bugs from model quality
- enable early testing without a trained model

### Why it is not versioned
- models are large binaries
- real models are user- or project-specific
- the repository defines the interface, not the trained weights

---

## ONNX Model Contract

### Input
- **Name**: `logmel` (configurable)
- **Shape**: `(1, n_mels, frames)`
- **Dtype**: `float32`
- **Content**: log-mel spectrogram

### Output
Supported output formats:
- `(1,)` → wake probability
- `(1, 1)` → wake probability (scalar wrapped in batch and channel)
- `(1, 2)` → `[not_wake, wake]` scores

The detector collapses all supported formats to a single scalar probability internally.

---

## Model Generation Constraints

ONNX models must be generated with:
- **IR version ≤ 11** (recommended: 7)
- **Opset ≤ 11**

These constraints are required for compatibility with ARM builds of ONNXRuntime
used on Raspberry Pi Zero 2 W.

Modern PyTorch exporters may produce higher opset models by default.
The export pipeline explicitly rewrites IR and opset versions after export.

---

## Dummy Model Generation Script

A minimal script is provided under `tools/` to generate a compatible dummy model.

Key properties:
- fixed output probability
- explicit IR and opset
- no dynamic shapes
- no unsupported operators

This model is used only for smoke testing the runtime.

---

## Bringing Real Training Data into Google Colab

Wake-word training is performed on a PC or in Google Colab.

### Uploading the dataset
1. Prepare the dataset directory locally:
   ```
   dataset/
     wake/
     non_wake/
   ```

2. Compress it:
   ```bash
   zip -r dataset.zip dataset
   ```

3. Upload `dataset.zip` to Google Drive.

### Mounting Google Drive in Colab
In Colab, mount Drive:
```python
from google.colab import drive
drive.mount("/content/drive")
```

Then copy and extract the dataset:
```python
import shutil
shutil.copy(
    "/content/drive/MyDrive/wakeword-training/dataset.zip",
    "/content/dataset.zip",
)

!unzip dataset.zip
```

The training notebook auto-detects the dataset root, even if it extracts as
`dataset/dataset`.

---

## Training Notebook

The complete training, export, and validation pipeline is documented in:

```
docs/dev/colab_wakeword_training_notebook.md
```

That document includes:
- dataset loading and validation
- TinyDSCNN training
- ONNX export compatible with ARM
- ONNXRuntime-based validation (not `onnx.checker`)

---

## Summary

- The repository defines the wake-word **interface**, not the trained weights
- ONNXRuntime is the final authority for model validity
- Output shapes `(1,)`, `(1,1)` and `(1,2)` are all supported
- Training and deployment are cleanly separated

This document is intended to remain stable across model iterations.

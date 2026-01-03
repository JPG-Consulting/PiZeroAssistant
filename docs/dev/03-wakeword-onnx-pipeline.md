# Wake-word ONNX Pipeline (Final)

This document describes the complete, reproducible pipeline used to build,
validate, train, export, and deploy an offline wake-word detector using ONNX.

It is hardware-agnostic and does not assume any local filesystem layout.

---

## 1. Dataset Structure

The dataset is expected to follow this structure:

    dataset/
    ├── wake/
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── ...
    └── not_wake/
        ├── noise_001.wav
        ├── speech_001.wav
        └── ...

All WAV files must be:
- Mono
- 16 kHz
- PCM (int16 or float32)

---

## 2. WAV Validation

Before training, validate audio integrity:

    python tools/validate_wavs.py <path/to/dataset/wake>
    python tools/validate_wavs.py <path/to/dataset/not_wake>

This checks:
- readable WAV headers
- correct sample rate
- mono audio

---

## 3. RMS & Clipping Audit

Next, audit signal quality:

    python tools/audit_rms.py <path/to/dataset/wake>
    python tools/audit_rms.py <path/to/dataset/not_wake>

Files may be flagged as:
- LOW_RMS (too quiet)
- HIGH_RMS
- CLIPPING

Flagged files are automatically moved into a `_bad/` subfolder and excluded from training.

---

## 4. Creating the Training ZIP

Create a ZIP containing the cleaned dataset folder.

Important:
- the folder **inside the ZIP** must be named `dataset/`
- the ZIP filename itself is arbitrary

Example:

    zip -r dataset.zip dataset

---

## 5. Google Colab Setup

In Google Colab, mount Drive:

    from google.colab import drive
    drive.mount("/content/drive")

Copy and extract:

    !cp /content/drive/MyDrive/dataset.zip .
    !unzip dataset.zip

Expected path after extraction:

    DATASET_ROOT = "/content/dataset"

---

## 6. Feature Extraction

The model uses:
- Log-Mel spectrograms
- 40 mel bins
- 25 ms window
- 10 ms hop
- 1.0 s context window

These parameters **must match deployment**.

---

## 7. Model Training

A TinyDSCNN-style binary classifier is trained using BCEWithLogitsLoss.

Class imbalance is handled using `pos_weight`.

---

## 8. ONNX Export

Export with:
- opset <= 11
- static input shape

---

## 9. Deployment Contract

### Input

    (1, 1, n_mels, frames) float32

### Output

    (1,) wake probability

---

This document represents the final, stable wake-word pipeline.

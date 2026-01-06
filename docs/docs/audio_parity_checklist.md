# Audio Parity Checklist
Recording ↔ Training ↔ Runtime

This document defines the invariants that must hold between:
- `tools/record_wakeword.py`
- `tools/training/train_wakeword.py`
- `src/assistant/wakeword/onnx_detector.py`

Any change violating these rules risks silent accuracy regression.

---

## 1. Audio capture invariants (MUST match)

| Property | Value | Enforced by |
|--------|------|------------|
| Sample rate | 16 kHz | ALSA + config |
| Channels | Mono (1) | record_wakeword.py |
| PCM format | int16 | ALSA |
| Device path | ALSA (same device) | record + runtime |
| Block size | config-driven | ALSA stream |

❌ No resampling at runtime  
❌ No normalization at recording time  

---

## 2. Recording (`tools/record_wakeword.py`)

### What it DOES
- Captures raw int16 PCM from ALSA
- Writes mono 16-bit WAV
- Applies **short fade-in/out only** (anti-click)

### What it MUST NOT do
- ❌ Apply gain
- ❌ Apply limiter
- ❌ Normalize audio
- ❌ Modify RMS intentionally

> Recording output represents *raw microphone capture*.

---

## 3. Runtime preprocessing (`OnnxWakeWordDetector`)

Order is **strict**:

1. Receive `int16` PCM from ALSA
2. Multiply by **2.5× fixed software gain**
3. Apply **tanh soft limiter** around ±20000
4. Cast back to `int16`
5. Extract log-mel features

This is the **reference preprocessing path**.

---

## 4. Training preprocessing (`train_wakeword.py`)

Training MUST mirror runtime exactly.

### Required steps (default path):
1. Load WAV → float32
2. Resample if needed
3. Pad / random-crop to `clip_seconds`
4. Convert to int16 PCM
5. Apply **2.5× gain**
6. Apply **tanh soft limiter**
7. Pass int16 to `LogMelExtractor`

### Allowed deviations
- `--no-limiter` flag (ABRATION ONLY)
- Random crop for augmentation

### Forbidden deviations
- ❌ Normalization
- ❌ RMS scaling
- ❌ Feature-side gain changes

---

## 5. Log-mel geometry invariants

These must match between training and runtime:

| Parameter | Source |
|---------|-------|
| `n_mels` | config.yaml |
| `n_fft` | config.yaml |
| `win_ms` | config.yaml |
| `hop_ms` | config.yaml |
| `clip_seconds` | config.yaml |

Use `--debug-logmel-shape` during training to enforce.

---

## 6. Safe change checklist (PR checklist)

Before merging audio-related changes:

- [ ] record_wakeword.py still writes raw int16
- [ ] runtime preprocessing unchanged
- [ ] training mirrors runtime preprocessing
- [ ] config.yaml unchanged or intentionally versioned
- [ ] log-mel shapes verified
- [ ] new model tested with `gold_test_onnx.py`

---

## 7. Golden rule

> **Recording is raw. Runtime defines truth. Training mirrors runtime.**

If unsure, inspect runtime code first.

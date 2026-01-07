# Audio Parity Contract

This document defines the **non-negotiable audio invariants** between:

- Recording (`tools/record_wakeword.py`)
- Training (`tools/training/train_wakeword.py`)
- Runtime inference (`src/assistant/audio/*`, `src/assistant/wakeword/*`, `src/assistant/dsp/*`)

Any change that violates this contract risks silent accuracy regressions.

The wake-word ONNX model contract is documented separately in:
- `03-wakeword-onnx-pipeline.md`

---

## Contract scope

This contract applies to the **ONNX wake-word pipeline**. The OpenWakeWord path and WebRTC VAD path are **not yet parity-validated** and must be treated as experimental.

---

## MUST match (recording → training → runtime)

### Audio format and capture
- **Mono, 16-bit PCM** only (`int16`, little-endian). Any other dtype or channel count is invalid.
- **Sample rate** must match `config/config.yaml` `audio.sample_rate` everywhere (record, train, runtime).
- **Block size** (`audio.block_ms`) defines the runtime frame size used by VAD and ALSA buffering.
  - If `vad.type=webrtc`, `block_ms` must be 10/20/30 ms and the sample rate must be 8/16/32/48 kHz.

### Feature extraction and clip geometry
- **Log-mel parameters** must match between training and runtime:
  - `features.n_fft`, `features.win_ms`, `features.hop_ms`, `features.n_mels`, `features.fmin`, `features.fmax`, `features.log_eps`.
- **Clip length** (`features.clip_seconds`) must be identical across training and runtime.
  - Runtime uses a ring buffer and pulls exactly `sample_rate * clip_seconds` samples.
  - Training trims/pads every clip to the same exact sample count.

### Preprocessing before LogMelExtractor (ONNX path)
- **Scaling and limiter must match exactly:**
  1. Convert samples to `int16` PCM.
  2. Apply a fixed **2.5x software gain**.
  3. Apply a **tanh soft limiter** around ±20000.
  4. Pass the limited `int16` samples into `LogMelExtractor`.
- Training mirrors the runtime path by default. Disabling the limiter (`--no-limiter`) is **ablation only** and breaks parity with runtime.

---

## MUST NOT change silently

The following are contract-critical and require retraining + explicit documentation updates:

- `audio.sample_rate`, `audio.channels`, `audio.dtype`
- `features.clip_seconds`, `features.n_mels`, `features.win_ms`, `features.hop_ms`, `features.n_fft`, `features.fmin`, `features.fmax`, `features.log_eps`
- Runtime preprocessing constants: **2.5x gain** and **tanh limiter at ±20000**
- Log-mel input shape expectations: **(1, 1, n_mels, frames)** in the ONNX model

---

## Known gaps / not parity-validated

- **WebRTC VAD:** Implemented but not fully tested in real usage. Audio/VAD behavior is **not yet parity-validated**.
- **OpenWakeWord detector:** Uses its own model and normalizes audio to float32 without the ONNX gain/limiter path. **Not parity-validated** against the ONNX training pipeline.

---

## Notes

- `record_wakeword.py` applies a short fade-in/out to avoid clicks; this should be preserved.
- Training resamples and trims/pads WAVs to enforce the runtime clip geometry. These steps are part of the parity contract.

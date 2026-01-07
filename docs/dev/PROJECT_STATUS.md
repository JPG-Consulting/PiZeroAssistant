# Project Status — PiZeroAssistant

Last updated: 2026-01-06

This document captures the **current state, frozen contracts, and next steps**
for the PiZeroAssistant project so work can be resumed after long gaps
or in new sessions.

---

## ✅ Stable & Frozen (Do Not Change Lightly)

### Wake-word runtime contract
- Input: `logmel` float32 `(1, 1, n_mels, frames)`
- Output: `prob` float32 `(1, 1)` in `[0, 1]`
- Frame count is **derived from config**, never hardcoded
- Preprocessing path is authoritative in:
  `src/assistant/wakeword/onnx_detector.py`

### ONNX export
- Opset: **18**
- Static batch size: **1**
- Sigmoid applied **inside ONNX**
- Runtime must NOT apply sigmoid again

### Audio parity
- Recording is raw int16 PCM (no gain, no normalization)
- Runtime defines truth
- Training mirrors runtime exactly
- See: `docs/audio_parity_checklist.md`

---

## 🧱 Architecture Snapshot

- Runtime: `src/assistant`
- Wake-word training: `tools/training/train_wakeword.py`
- Feature extraction: shared `LogMelExtractor`
- Models live outside Git; repo defines the **interface**, not weights

---

## 🗑️ Explicitly Deprecated / Legacy

- `docs/dev/colab_wakeword_training_notebook.md` (deleted)
- Opset 11 references exist **only** for:
  - legacy / dummy models
  - inference-path testing (`make_dummy_wake_onnx.py`)

These are **not** the runtime contract.

---

## 🔜 Planned / Suggested Next Steps (Not Started)

- [ ] Optional: add CI guard against hardcoded frame counts (e.g. `98`)
- [ ] Optional: move legacy ONNX dummy tools under `tools/legacy/`
- [ ] Optional: add ONNX runtime smoke-test script to CI
- [ ] Optional: evaluate INT8 quantization for Pi Zero 2 W
- [ ] Optional: document wake-word threshold calibration workflow

---

## 🚫 Things to Avoid

- Reintroducing fixed frame counts in docs or code
- Mixing Colab tutorials into contract documentation
- Changing preprocessing order without updating *all* parity docs

---

## 📍 Where to Resume Work

Recommended next focus area:
> (fill in when you decide — e.g. threshold tuning, quantization, metrics, etc.)

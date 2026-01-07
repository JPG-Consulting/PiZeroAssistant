# Audio Parity Contract

This document defines the **non-negotiable audio invariants** between:

- Recording (`tools/record_wakeword.py`)
- Training (`tools/training/train_wakeword.py`)
- Runtime inference (`src/assistant`)

Any change that violates this contract risks silent accuracy regressions.

This document is the authoritative reference for:
- Sample format
- Gain / limiter behavior
- Feature extraction alignment

The wake-word ONNX contract is documented separately in:
- `03-wakeword-onnx-pipeline.md`

# Models

This folder is the **local drop location** for wake-word models and related runtime artifacts.

✅ This repository **tracks this README**  
🚫 This repository **does not track model binaries** (`.onnx`, `.onnx.data`)

---

## What goes in here

At runtime, the assistant loads an ONNX wake-word model from this folder (via `config/config.yaml`).

Typical contents:

```
models/
├── README.md
├── wakeword.onnx
└── wakeword.onnx.data
```

Notes:
- `wakeword.onnx.data` is created by the exporter for **external tensor data** (large weights stored separately).
- Both files must be present together when external data is used.

---

## What is tracked vs ignored

Tracked:
- `models/README.md` (this file)

Ignored (not committed):
- `models/*.onnx`
- `models/*.onnx.data`
- Any other model weights or generated artifacts

Rationale:
- Model binaries are large and frequently retrained.
- Different environments may require different models.
- The repository defines the **interface/contract**, not the trained weights.

---

## Naming & versioning (recommended)

You may keep the default filename `wakeword.onnx` for runtime simplicity, but you should still
**version your exported artifacts** when archiving or sharing them.

Recommended naming for archived models:

```
wakeword_<name>_vMAJOR.MINOR.PATCH.onnx
wakeword_<name>_vMAJOR.MINOR.PATCH.onnx.data
```

Example:

```
wakeword_jarvis_v1.0.0.onnx
wakeword_jarvis_v1.0.0.onnx.data
```

### Version meaning (Semantic Versioning)

- **MAJOR**: breaking change to the model contract (e.g., input shape, features, output semantics)
- **MINOR**: new training run / improvements without changing the contract
- **PATCH**: small fix (e.g., calibration tweak, minor data cleanup) without changing the contract

---

## Contract reminder (must match runtime)

Your exported model must match the assistant’s feature extractor settings:

- Sample rate: **16 kHz**
- Log-mel bins: **40**
- Window / hop: **25 ms / 10 ms**
- Context window: **1.0 s**
- Input tensor name: usually **`logmel`**
- Input shape: **(1, 1, n_mels, frames)** `float32` where frames are derived from config
- Output name: usually **`prob`**
- Output shape: **(1, 1)** `float32` probability in `[0,1]`

If these do not match, detection will fail or behave unpredictably. Frame counts must not be hardcoded.

---

## How to use a newly trained model

1. Export from Colab as:
   - `wakeword.onnx`
   - `wakeword.onnx.data` (if produced)

2. Copy both files into:
   - `models/`

3. Point the runtime config at it:
   - `wakeword.model_path: models/wakeword.onnx`

4. Run the assistant and verify output with your local tools.

---

## Tip: keep an archive outside Git

Keep a separate folder outside the repo (or in Google Drive) as your model registry, e.g.:

```
wakeword-training/
├── exports/
│   ├── wakeword_jarvis_v1.0.0.onnx
│   ├── wakeword_jarvis_v1.0.0.onnx.data
│   └── notes.md
```

Then copy the chosen “current” model into `models/` as `wakeword.onnx` for deployment.

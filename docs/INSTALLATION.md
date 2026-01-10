# Installation

## Requirements

- **Python 3.11** is required due to `openwakeword==0.5.1` compatibility.
- Raspberry Pi OS **Bookworm** includes Python 3.11.
- Raspberry Pi OS **Trixie** ships with Python 3.13, so Python 3.11 must be installed manually.

For platform policy and safe installation steps, see `docs/dev/python-versions.md`.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Installing a pretrained openWakeWord model

The assistant will not start without a wake-word model. This project does not include a model by default. You must download a pretrained openWakeWord model manually and place it on disk.

Download a model from the openWakeWord GitHub Releases page (https://github.com/dscripka/openWakeWord). Example commands:

```bash
mkdir -p models
cd models

# ONNX model
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx

# OR TFLite model
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite
```

ONNX is typically used for CPU inference on x86 systems. TFLite may be preferred on more constrained devices. This project does not guarantee performance for any specific model or backend.

Configure the model path in your YAML config:

```yaml
wakeword:
  model_path: "models/hey_jarvis_v0.1.onnx"
```

The runtime selects the backend based on the file extension: `.onnx` uses ONNXRuntime and `.tflite` uses TFLite. There is no auto-conversion or runtime downloading.

If the model file is missing or unreadable, startup fails with a clear error and the wake-word service does not start. This is intentional and not a bug.

Training custom wake words is out of scope for this installation guide. For advanced model creation or customization, refer to developer documentation instead.

Then run:

```bash
python -m voiceassistant.main --config config/config.yaml
```

# Installation

## Requirements

- **Python 3.11** is required due to `openwakeword==0.5.1` compatibility.
- Raspberry Pi OS **Bookworm** includes Python 3.11.
- Raspberry Pi OS **Trixie** ships with Python 3.13, so Python 3.11 must be installed manually.

Installation assumes sudo access to write to `/opt` and to run `ldconfig` when required by the platform.
For Raspberry Pi OS Trixie and Pi Zero / Zero 2 details, see `docs/dev/python-versions.md`.

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

**Wakeword model format note:** The wakeword model format must match the configured inference backend.
For Raspberry Pi Zero / Zero 2, TFLite (`.tflite`) models are recommended.

The runtime selects the backend based on the file extension: `.onnx` uses ONNXRuntime and `.tflite` uses TFLite. There is no auto-conversion or runtime downloading.

If the model file is missing or unreadable, startup fails with a clear error and the wake-word service does not start. This is intentional and not a bug.

Training custom wake words is out of scope for this installation guide. For advanced model creation or customization, refer to developer documentation instead.

This project uses a `src/` layout, so Python needs to be told where to find the `voiceassistant` package. This is expected and intentional during development.

Then run:

```bash
PYTHONPATH=src python -m voiceassistant.main --config config/config.yaml
```

Advanced users can optionally install the project in editable mode to avoid setting `PYTHONPATH`, but it is not required to get started.

## Supported execution modes

- Development without installation: requires `PYTHONPATH=src`.
- Development with an editable install in a virtual environment: no `PYTHONPATH` required; the `voiceassistant` console script is available.
- Service / production execution: use the virtual environment interpreter or console script; `PYTHONPATH` must not be relied upon in system services.

## Optional conversation memory persistence

Conversation memory is in-memory by default. To enable local persistence, set:

```yaml
conversation:
  persistence:
    enabled: true
    path: "/var/lib/voiceassistant/conversation.json"
```

The file is stored locally on disk at the configured path and is never transmitted. To disable persistence, set `conversation.persistence.enabled` back to `false` (or remove the stanza). When disabled, no disk I/O occurs for conversation history.

## Optional: ReSpeaker Pi HAT LED support

LED support is optional. The ReSpeaker Pi HAT (APA102) LEDs use the Raspberry Pi SPI interface via native `spidev`, with best-effort behavior that automatically falls back to a no-op backend if SPI or `spidev` is unavailable. No configuration flags are required.

Enable SPI, then reboot:

```bash
sudo raspi-config
# Interface Options → SPI → Enable
sudo reboot
```

On Raspberry Pi OS, `spidev` is usually available by default. If it is missing, install it via:

```bash
sudo apt install python3-spidev
```

If SPI or `spidev` is unavailable, the assistant still runs normally; LEDs remain off and the backend logs a warning.

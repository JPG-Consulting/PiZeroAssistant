# How to Run the Wake Word Assistant (Raspberry Pi Zero 2 W)

This guide explains how to **set up**, **run**, and **re-run** the wake word detection system using a Python virtual environment (`venv`).

The assistant currently supports:
- ALSA microphone input (via PortAudio / `sounddevice`)
- Offline VAD
- Offline ONNX wake word detection
- Optional OpenWakeWord backend (set `wakeword.type: openwakeword`)

It is designed to be **production-ready** and easy to update from GitHub.

---

## Prerequisites

### System packages (Raspberry Pi OS)
Make sure the following system dependencies are installed:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip                     portaudio19-dev libatlas-base-dev
```

(Optional but recommended for WebRTC VAD)
```bash
sudo apt install -y build-essential
```

---

## First-time setup

Run the following commands **from the root of the repository**.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

This creates an isolated Python environment in the `.venv` directory so system Python packages are not affected.

---

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

After activation, your shell prompt should change and show something like:

```text
(.venv) pi@raspberrypi:~/your-repo $
```

---

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs all required libraries:
- `onnxruntime`
- `sounddevice`
- `webrtcvad` (optional)
- `numpy`, `scipy`, `PyYAML`, etc.

If you want to try the **OpenWakeWord** backend, also install the optional package:

```bash
pip install openwakeword
```

---

### 4. Run the assistant

```bash
# from repo root:
PYTHONPATH=src python -m assistant.main --config config/config.yaml
```

What this does:
- `PYTHONPATH=src` allows Python to find the `assistant` package
- `assistant.main` is the runtime entrypoint
- `config/config.yaml` controls audio, VAD, and wake word parameters

You should see log output similar to:

```text
[INFO] VAD: WebRTC (aggressiveness=2)
[INFO] Wakeword model: models/wakeword_tiny_dscnn_int8.onnx
[INFO] Listening...
```

Say your wake word near the microphone to test detection.

Press **Ctrl+C** to stop.

---

## Re-running after a `git pull`

When you update the repository with new code:

```bash
git pull
```

### Typical case (most common)
If **only Python source files changed**, simply run:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m assistant.main --config config/config.yaml
```

No reinstallation is needed.

---

### If `requirements.txt` changed
If new Python dependencies were added or versions changed:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Then re-run:

```bash
PYTHONPATH=src python -m assistant.main --config config/config.yaml
```

---

### If the virtual environment was deleted
If `.venv` was removed or corrupted, recreate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m assistant.main --config config/config.yaml
```

---

## Common tips

### Microphone device issues
To list available audio devices:

```bash
python - << 'EOF'
import sounddevice as sd
print(sd.query_devices())
EOF
```

Then set the desired device index or name in `config/config.yaml`:

```yaml
audio:
  device: 2
```

---

## Running on boot (later)
For production deployment, this assistant can later be:
- Wrapped in a `systemd` service
- Auto-started on boot
- Run headless with logs redirected to a file

(This is intentionally **not** enabled yet during development.)

---

## Summary

### First time:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m assistant.main --config config/config.yaml
```

### After updates:
```bash
git pull
source .venv/bin/activate
pip install -r requirements.txt   # only if needed
PYTHONPATH=src python -m assistant.main --config config/config.yaml
```

You now have a clean, reproducible way to run the wake word assistant as a real product.

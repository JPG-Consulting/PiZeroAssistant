# PiZeroAssistant

PiZeroAssistant is an offline wake-word listener built for very small Linux boards such as the Raspberry Pi Zero 2 W. It listens for a specific phrase using a compact ONNX model and triggers local actions without sending audio to the cloud.

## Why it exists
- Provide a privacy-preserving wake-word experience that keeps audio on-device.
- Run reliably on low-power hardware with limited CPU and memory.
- Offer a simple, hackable baseline for experimenting with embedded voice triggers.

## How it works (high level)
1. **Audio capture:** ALSA records mono, 16-bit PCM at 16 kHz from the selected microphone.
2. **Feature extraction:** Short windows are converted into log-mel spectrograms via `assistant.dsp.LogMelExtractor`.
3. **Wake-word model:** A lightweight ONNX model consumes the features and estimates whether the wake word is present.
4. **Trigger:** When the score crosses the configured threshold, a wake event is emitted for downstream handling.

## Design focus
- **Offline:** No internet connection is required for detection.
- **Privacy-preserving:** Audio never leaves the device; processing stays local.
- **Low-resource friendly:** Tuned for constrained boards and modest microphones.

## Repository overview
- `src/assistant/` — Runtime code for audio I/O, feature extraction, VAD, metrics, and wake-word inference.
- `config/config.yaml` — Central configuration for audio settings, feature parameters, VAD choice, and model paths.
- `systemd/` — Service templates for running the assistant as a background daemon.
- `scripts/` — Helper scripts for system setup and service installation.
- `tools/` — Utilities for recording, labeling, and validating wake-word datasets and models.
- `models/` — Placeholder for bundled or downloaded ONNX wake-word models.
- `docs/` — Additional reference material.

## Running the assistant
The recommended way to keep the wake listener active is via the provided `systemd` unit (`assistant.service.in`), which runs the module entrypoint with the configured environment. For quick experiments or debugging, you can invoke the Python module directly:

```
python -m assistant.main --config config/config.yaml
```

Adjust the configuration file to match your microphone device, VAD choice, and model path before running.

## What this project is not
- Not a general-purpose voice assistant with cloud skills or NLU.
- Not a speech recognition or dictation system.
- Not a hosted service; all inference stays on the local device.

## Working with datasets and models
The `tools/` directory contains small command-line utilities to help collect audio, curate wake-word clips, and verify ONNX models before deploying them to constrained boards. These tools are separate from the runtime and are meant for offline preparation.

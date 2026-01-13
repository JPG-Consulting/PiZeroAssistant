# Voice Assistant

Voice Assistant is a production-focused, offline-first voice assistant tailored for the Raspberry Pi Zero / Zero 2 W. The system is designed to run continuously with deterministic behavior, bounded resource usage, and explicit recovery semantics.

## Highlights

- Always-on offline wake-word detection (openWakeWord + WebRTC VAD)
- Event-driven pipeline with explicit state transitions
- Record-then-send STT and LLM calls with provider routing
- Interruptible TTS playback with true barge-in
- Strict buffering and bounded queues for real-time safety

## Quick start

1. Install dependencies (Raspberry Pi OS Lite, 32-bit recommended):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy the example configuration and edit it:

```bash
cp config/config.example.yaml config/config.yaml
```

3. Run the assistant:

```bash
PYTHONPATH=src python -m voiceassistant.main --config config/config.yaml
```

Optional: if you install the project in a virtual environment (for example, `pip install -e .`), a console script named `voiceassistant` becomes available and can be used instead:

```bash
voiceassistant --config config/config.yaml
```

The console script is only available after installation and is optional during development.

### Type checking

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m mypy
```

### Python version requirement

Python 3.11 is required due to `openwakeword==0.5.1`. Raspberry Pi OS Bookworm includes Python 3.11, while Raspberry Pi OS Trixie ships with Python 3.13 and needs a separate Python 3.11 install. See `docs/INSTALLATION.md` for setup steps and `docs/dev/python-versions.md` for platform details.

## Configuration

Configuration is YAML-based. Secrets must be provided via environment variables referenced in the config (`api_key_env`). See `config/config.example.yaml` for all available options.

### Provider API contract (HTTP)

The default HTTP providers expect the following payloads and responses:

- **STT** (`POST /stt`): multipart form upload with `file` (audio/wav). Response JSON: `{ "text": "..." }`.
- **LLM** (`POST /llm`): JSON `{ "prompt": "..." }`. Response JSON: `{ "text": "..." }`.
- **TTS** (`POST /tts`): JSON `{ "text": "..." }`. Response either audio bytes with `Content-Type: audio/wav` or JSON `{ "audio_wav_base64": "..." }`.
See `DEVELOPMENT.md` for the authoritative STT provider contract, including handling of empty transcripts.

## Project layout

- `src/voiceassistant/` — core assistant code
- `config/` — configuration examples
- `tools/` — minimal test utilities
- `DEVELOPMENT.md` — architecture and behavioral contracts

## License

MIT. See `LICENSE`.

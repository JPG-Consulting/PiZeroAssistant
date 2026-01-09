# Installation

## Requirements

- **Python 3.11** is required due to `openwakeword>=0.5.0` compatibility.
- Raspberry Pi OS **Bookworm** includes Python 3.11.
- Raspberry Pi OS **Trixie** ships with Python 3.13 and requires a separate Python 3.11 install.

For details and safe installation steps on Trixie, see `docs/dev/python-versions.md`.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run:

```bash
python -m voiceassistant.main --config config/config.yaml
```

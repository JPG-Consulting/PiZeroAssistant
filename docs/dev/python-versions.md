# Python versions

## Overview

Pi Zero Voice Assistant currently requires **Python 3.11** due to dependency constraints in `openwakeword>=0.5.0`. This is a platform constraint, not a preference. The guidance below explains the supported versions and how to install Python 3.11 safely when needed.

## openWakeWord and Python compatibility

- `openwakeword>=0.5.0` supports Python 3.11.
- Python 3.13 is not supported by `openwakeword` at this time.

## Raspberry Pi OS Python versions

- **Bookworm** ships with Python 3.11 (supported).
- **Trixie** ships with Python 3.13 (not supported for `openwakeword`).

## Why Python 3.11 is required today

The wakeword service uses `openwakeword>=0.5.0`. Until `openwakeword` officially supports Python 3.13, the assistant must run on Python 3.11 to ensure reliable wakeword detection and package compatibility.

## Installing Python 3.11 on Raspberry Pi OS Trixie

> Do not replace the system Python. Install Python 3.11 alongside it and use a dedicated virtual environment.

1. Install build and packaging prerequisites:

   ```bash
   sudo apt update
   sudo apt install -y build-essential libssl-dev zlib1g-dev \
     libncurses5-dev libncursesw5-dev libreadline-dev \
     libsqlite3-dev libffi-dev libbz2-dev liblzma-dev \
     tk-dev wget
   ```

2. Download and build Python 3.11:

   ```bash
   cd /tmp
   wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
   tar -xzf Python-3.11.9.tgz
   cd Python-3.11.9
   ./configure --enable-optimizations
   make -j$(nproc)
   sudo make altinstall
   ```

   `make altinstall` ensures Python 3.11 is installed alongside the system Python.

3. Create and activate a Python 3.11 virtual environment:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

4. Verify the active Python version:

   ```bash
   python --version
   ```

   The output should read `Python 3.11.x`.

## Future expectations (non-binding)

Python version requirements will be revisited when `openwakeword` adds support for newer Python releases. Until then, Python 3.11 remains the required version for production builds.

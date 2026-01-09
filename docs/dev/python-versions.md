# Python versions

## Overview

Pi Zero Voice Assistant currently requires **Python 3.11** due to dependency constraints in `openwakeword==0.5.1`. This is a platform constraint, not a preference. The guidance below explains the supported versions and how to install Python 3.11 safely when needed.

## openWakeWord and Python compatibility

- `openwakeword==0.5.1` supports Python 3.11.
- Python 3.13 is not supported by `openwakeword` at this time.

## Raspberry Pi OS Python versions

- **Bookworm** ships with Python 3.11 (supported).
- **Trixie** ships with Python 3.13 (not supported for `openwakeword`).

## Why Python 3.11 is required today

The wakeword service uses `openwakeword==0.5.1`. Until `openwakeword` officially supports Python 3.13, the assistant must run on Python 3.11 to ensure reliable wakeword detection and package compatibility.

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

2. Download and build Python 3.11 (safe settings for low-memory devices):

   ```bash
   cd /tmp
   wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
   tar -xzf Python-3.11.9.tgz
   cd Python-3.11.9
   ./configure \
     --prefix=/usr/local/python3.11 \
     --enable-shared \
     --without-ensurepip
   make -j1
   sudo make altinstall
   ```

   **Warning:** Do not use `--enable-optimizations`, PGO, LTO, or `make -j$(nproc)` on Pi
   Zero / Zero 2. These options are known to fail on low-memory devices and can crash the
   build toolchain. A single-threaded build without PGO/LTO is slower but reliable and is
   the supported approach for this project.

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

### Notes for Raspberry Pi Zero / Zero 2

Python is expensive to build on embedded hardware. The Pi Zero 2W has limited RAM, and
aggressive build flags (PGO, LTO, or high parallelism) reliably exhaust memory. For this
project, reliability is prioritized over build speed or runtime performance. A slow,
single-threaded build is expected and acceptable on these devices.

### Known build failures on low-memory devices

On Pi Zero / Zero 2-class hardware, the following failures are repeatable when PGO/LTO or
high parallelism is enabled:

- `gcc: fatal error: Killed signal terminated program cc1`
- Linker crashes during PGO stages (e.g., `profile-gen-stamp`)
- `__gcov_indirect_call` undefined references during linking
- Segmentation faults in `ld` or `collect2`

These failures are caused by insufficient RAM, PGO profiling passes, and excessive parallel
compilation. They are not Python bugs and not user mistakes. Retrying with the same flags
will not succeed on constrained hardware. Use the safe, single-threaded build described
above instead.

### Alternative workflow: build elsewhere and copy

If local builds are still too slow or unreliable, you can build Python 3.11 on a more
powerful system and copy the installation to the Pi Zero / Zero 2. This is optional,
advanced, and intended for users who hit local build limits.

High-level procedure:

1. Build Python 3.11 on a compatible ARM Linux system (e.g., Pi 4 / Pi 5) using a custom
   prefix such as `/opt/python3.11` (do not overwrite system Python).
2. Copy the resulting directory to the target device (e.g., `rsync` or `scp`).
3. Use the copied interpreter to create virtual environments on the target device.

Constraints and caveats:

- Build and target architectures must be compatible.
- System libraries should match closely between build and target devices.
- Do not replace the system Python.
- This workflow assumes basic Linux/ARM familiarity.

## Future expectations (non-binding)

Python version requirements will be revisited when `openwakeword` adds support for newer Python releases. Until then, Python 3.11 remains the required version for production builds.

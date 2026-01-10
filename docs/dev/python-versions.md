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

### Building Python 3.11 on another Raspberry Pi and copying it

On Raspberry Pi Zero / Zero 2–class hardware, compiling Python 3.11 locally can be very slow
and, in some cases, unreliable. An alternative and often preferable approach is to build
Python 3.11 on a more powerful Raspberry Pi (such as a Pi 4 or Pi 5) and copy the resulting
installation to the target device.

This workflow is supported when the build and target systems run the same Raspberry Pi OS
release, use the same architecture and bitness, and the Python installation is kept
isolated from the system Python.

#### Step 1: Build on the build machine

Install dependencies:

```bash
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
  libncurses5-dev libncursesw5-dev libreadline-dev \
  libsqlite3-dev libffi-dev libbz2-dev liblzma-dev \
  tk-dev wget
```

Download and configure Python:

```bash
cd /tmp
wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
tar -xzf Python-3.11.9.tgz
cd Python-3.11.9
./configure \
  --prefix=/opt/python3.11 \
  --enable-shared \
  --without-ensurepip
```

Build and install:

```bash
make -j$(nproc)
sudo make install
```

#### Step 2: Copy to the target device

```bash
rsync -a \
  --rsync-path="sudo rsync" \
  /opt/python3.11 \
  youruser@PI_ZERO_IP:/opt/
```

`/opt` is root-owned on the target device. If you run `rsync` without `sudo`, it will fail
with `Permission denied`, which is expected behavior because the remote side is executing
as your SSH user. The command above runs `sudo rsync` on the target device and will prompt
for your sudo password.

_Fallback (if `sudo rsync` is unavailable):_ copy into your home directory and move it with
`sudo`:

```bash
rsync -a /opt/python3.11 youruser@PI_ZERO_IP:~/
sudo mv /home/youruser/python3.11 /opt/
```

#### Verify and register shared libraries on the target device

Because Python was built with `--enable-shared`, the target device must register
`/opt/python3.11/lib` with the dynamic linker cache. This must be done on the target device,
even if it was already done on the build machine. You only need to do this once per device.

```bash
sudo sh -c 'echo /opt/python3.11/lib > /etc/ld.so.conf.d/python3.11.conf'
sudo ldconfig
```

Verify the interpreter immediately after copying:

```bash
/opt/python3.11/bin/python3.11 --version
```

If this command fails with `libpython3.11.so.1.0` not found, the linker step above was
missed. You can also confirm the shared library resolution with:

```bash
ldd /opt/python3.11/bin/python3.11 | grep libpython
```

#### Step 3: Use on the target device

```bash
/opt/python3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
```

### Recommendation

For Raspberry Pi Zero / Zero 2:

- **Preferred:** Build Python 3.11 on a Pi 4 / Pi 5 and copy it
- **Supported:** Local single-threaded build without PGO/LTO
- **Not supported:** Local PGO/LTO or aggressive parallel builds

## Future expectations (non-binding)

Python version requirements will be revisited when `openwakeword` adds support for newer
Python releases. Until then, Python 3.11 remains the required version for production builds.

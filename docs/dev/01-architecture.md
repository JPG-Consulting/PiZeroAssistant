# Architecture Overview

This document describes the high-level architecture of PiZeroAssistant.

It focuses on:
- Runtime components and their responsibilities
- Audio data flow from ALSA to wake-word detection
- Separation between recording, training, and inference

This is **not** a tutorial.

Detailed implementation notes live close to the code.
Wake-word model contracts are documented in:
- `03-wakeword-onnx-pipeline.md`

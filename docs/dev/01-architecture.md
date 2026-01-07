# Architecture Overview

This document describes the high-level architecture of PiZeroAssistant.

It focuses on:
- Runtime components and their responsibilities
- Audio data flow from ALSA to wake-word detection
- Component boundaries and validation status

This is **not** a tutorial.

Detailed implementation notes live close to the code.
Wake-word model contracts are documented in:
- `03-wakeword-onnx-pipeline.md`

---

## System boundaries

**In-repo (this repository):**
- Assistant runtime pipeline (audio I/O, VAD, wake-word detection)
- Contracts for model input/output and audio parity
- Stubs or interfaces for future STT/LLM/TTS integration

**Out of scope (external repositories/services):**
- OpenAI-compatible STT/LLM/TTS servers (e.g., Raspberry Pi 5 services, Piper)

---

## Component overview and status

| Component | Responsibility | Status |
| --- | --- | --- |
| Audio input (ALSA) | Capture mono PCM16 audio frames via `AlsaMicStream` | **Implemented & tested** |
| VAD (Energy) | Adaptive energy-based speech detection | **Implemented & tested** |
| VAD (WebRTC) | WebRTC VAD speech detection | **Implemented but not fully tested** (experimental) |
| Wake-word detector (ONNX) | Log-mel extraction + ONNX inference | **Implemented; under validation** |
| Wake-word detector (OpenWakeWord) | Uses `openwakeword` model inference | **Implemented but not fully tested** |
| Wake-word detector (Simulated) | Fake detector for pipeline testing | **Implemented & tested** |
| STT interface | Audio-to-text contract | **Planned / stubbed** (stub provider exists) |
| LLM interface | Text-to-text contract | **Planned / stubbed** |
| TTS interface | Text-to-speech contract | **Planned / stubbed** |

---

## Audio pipeline (runtime)

1. **Audio input (ALSA):** `AlsaMicStream` captures mono PCM16 frames at the configured sample rate.
2. **VAD:** One of:
   - **Energy VAD** (tested)
   - **WebRTC VAD** (implemented but not fully tested)
3. **Wake-word detection:**
   - **ONNX detector** (log-mel features + ONNX inference) — implemented, under validation.
   - **OpenWakeWord detector** — implemented, not fully tested.
   - **Simulated detector** — development/testing only.
4. **Future pipeline:** Wake-word triggers are intended to feed STT → LLM → TTS once those interfaces are integrated.

---

## Validation status labels

- **Implemented & tested:** Verified in real usage.
- **Implemented but not fully tested / Experimental:** Code exists but has not been validated in real usage.
- **Planned / stubbed:** Interfaces may exist, but runtime behavior is not yet implemented.

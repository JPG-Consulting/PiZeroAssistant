# Project Status — PiZeroAssistant

## Purpose of this document
This file is the **single source of truth for project continuity**.
It captures the current state, decisions, and next steps so development can be resumed even if chat history or context is lost.

---

## High-level goal

Offline-first voice assistant with:
- Wake Word Detection (offline)
- STT via configurable OpenAI-compatible API
- LLM via configurable OpenAI-compatible API
- TTS via configurable OpenAI-compatible API (Piper initially)

NOTE: The OpenAI-compatible server on Raspberry Pi 5 is a **separate repository**.

---

## Current focus

Phase 1 — Wake Word Detection (KWS)

Status:
- Training pipeline: stable
- Feature parity: stable
- ONNX export: opset 18, static batch=1
- Runtime inference: functional, under validation
- Threshold tuning: ongoing
- WebRTC VAD: not fully tested

---

## Next phases

Phase 2: STT integration  
Phase 3: LLM integration  
Phase 4: TTS integration  

All backends configurable via YAML.

---

## Documentation state

- PROJECT_STATUS.md: current
- docs/dev/03-wakeword-onnx-pipeline.md: current
- docs/dev/01-architecture.md: TODO
- docs/dev/02-audio-parity.md: TODO

---

Guiding rule: **Code defines truth; docs define contracts.**

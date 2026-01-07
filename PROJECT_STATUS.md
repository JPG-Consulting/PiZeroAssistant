# Project Status — PiZeroAssistant

## Purpose of this document
This file is the **single source of truth for project continuity**.
It captures the current state, decisions, and next steps so development can be resumed even if chat history or context is lost.

---

## High-level goal

Offline-first voice assistant with:
- Wake Word Detection (offline)
- STT via configurable OpenAI-compatible API (planned; interfaces will exist here)
- LLM via configurable OpenAI-compatible API (planned; interfaces will exist here)
- TTS via configurable OpenAI-compatible API (planned; interfaces will exist here)

NOTE: External OpenAI-compatible servers (e.g., Raspberry Pi 5 services or Piper) are **separate repositories** and are out of scope for this repo.

---

## Current focus (blocking milestone)

Phase 2 — Wake Word Detection (KWS)

KWS stability is the current blocking milestone. Runtime detection is implemented but still under validation.

---

## Phase status

**Phase 1 — Voice Activity Detection (VAD)**
- Energy-based VAD: implemented and tested
- WebRTC VAD: implemented but not fully tested (not yet production-validated)

**Phase 2 — Wake-Word Detection (KWS)** *(current focus)*
- Training pipeline: implemented
- Feature extraction parity: implemented (still under validation)
- ONNX export: implemented (opset 18, static batch=1)
- Runtime inference: implemented (under validation)
- Threshold tuning: ongoing

**Phase 3+ — STT → LLM → TTS** *(planned)*
- STT/LLM/TTS backends are intended to be configurable and OpenAI-API-compatible.
- Interfaces are planned or stubbed in this repo; production-grade backends live elsewhere.

---

## Documentation state

- PROJECT_STATUS.md: current
- docs/dev/01-architecture.md: current
- docs/dev/02-audio-parity.md: current
- docs/dev/03-wakeword-onnx-pipeline.md: current

---

Guiding rule: **Code defines truth; docs define contracts.**

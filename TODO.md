# TODO — Voice Assistant Roadmap

This document tracks concrete, incremental work items for the Voice Assistant project.
All items must respect the architectural invariants defined in DEVELOPMENT.md.

## Short-term (next implementation steps)

- [ ] **Conversation memory implementation**

  - In-memory conversational context owned by the state machine
  - Explicit reset commands (`"reset conversation"`)
  - No provider-side memory

- [ ] **Optional conversation persistence**

  - Disabled by default
  - Local-only storage
  - Clear privacy invariants

- [x] **Provider registry / factory**

  - Move provider construction out of `state_machine.py`
  - Allow adding new providers without touching core runtime logic

- [ ] **Provider capability documentation**

  - Document required interfaces for STT / LLM / TTS providers
  - Include minimal examples

- [ ] **Wake beep asset handling**

  - Document optional wake beep behavior
  - Fail gracefully if asset is missing

- [~] **Support compressed audio formats for STT** (metadata only; encoding/negotiation pending)

  - Allow sending OPUS and/or MP3 audio to STT providers
  - Keep WAV as the default for compatibility
  - Encoding must be explicit and configurable per provider

- [ ] **STT encoding selection**

  - Choose an encoding based on provider `audio_formats` and pipeline constraints
  - Keep WAV as the default fallback when no match exists

- [ ] **Per-provider audio format negotiation**

  - Select STT input format per provider during routing
  - Avoid implicit negotiation or silent fallback

- [ ] **Internationalize assistant TTS messages**

  - Central message catalog for assistant replies (keys like `conversation.reset_ack`)
  - Language selection mechanism (config- or provider-driven)
  - Explicit English fallback with clear errors if translations are missing
  - Tests that verify fallback-to-English behavior
  - Applies only to assistant-generated TTS output; user transcripts remain language-agnostic
  - Providers must not handle translation or localization; no cloud translation dependencies

## Medium-term (stability and usability)

- [ ] **Provider health introspection**

  - Log active provider and fallback reason at INFO level
  - Expose provider selection clearly in logs

- [ ] **Explicit user-facing diagnostics**

  - Optional spoken feedback on failures (“I’m having trouble connecting”)
  - Configurable verbosity

- [ ] **Improved barge-in handling**

  - Ensure playback interruption works under all load conditions

- [ ] **Config validation hardening**

  - Early validation for incompatible options
  - Clear, actionable error messages

## Long-term (optional, non-blocking)

- [ ] Streaming STT (opt-in, not default)
- [ ] Local-only LLM support
- [ ] Plugin-style provider loading (explicit, not automatic)

These are not prerequisites for a usable system.

## Explicit non-goals

- ❌ End-to-end ML training pipelines
- ❌ Provider auto-detection or silent fallback
- ❌ Hidden conversational memory
- ❌ Cloud dependency in idle mode
- ❌ Large frameworks or async-heavy architectures

## Architectural invariants (do not violate)

- Providers are stateless and synchronous.
- Conversational memory belongs to the state machine.
- Wakeword detection is always local and always on.
- Network calls only occur after a wake event.
- Failures must return the assistant to IDLE.

Any change that violates these rules must update DEVELOPMENT.md and be justified explicitly.

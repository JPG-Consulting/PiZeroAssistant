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

- [ ] **Unify provider endpoint semantics (full operation endpoints)**

  - Align all providers with the invariant that configuration specifies full, operation-specific HTTP endpoints.
  - Remove provider-owned path composition (providers must not append or modify endpoint paths).
  - Treat LAN-hosted and internet-hosted services uniformly as third-party APIs.
  - Update provider implementations and example configs accordingly.
  - This is a mechanical refactor only; no behavior change intended.
  - Must preserve all existing architectural invariants in DEVELOPMENT.md.

- [x] **Local echo LLM provider (diagnostic only)**

  - Returns the user message verbatim (or lightly formatted)
  - No external dependencies
  - No provider-side memory
  - Used for STT → LLM → TTS smoke testing
  - Safe to disable or remove later
  - Implements the same LLM provider interface
  - Selectable via `provider_type`
  - Must not affect routing, memory, or persistence semantics

- [ ] **End-to-end STT/TTS smoke test**

  - Use echo LLM provider
  - Verify spoken input is transcribed and spoken back
  - No dependency on real LLMs

- [ ] **LLM provider capability documentation**

  - Explicitly document statelessness invariant
  - Document that providers must not store chat history
  - Clarify difference between provider logic and state-machine-owned memory

- [ ] **LLM token budgeting (future)**

  - Per-provider rolling budgets (hour/day)
  - Router-enforced cooldown on budget exhaustion
  - Provider-reported vs estimated token accounting
  - Optional spoken UX feedback

- [x] **Skip LLM/TTS execution on empty STT transcript (token-safe behavior)**
- [x] **Documented STT empty-transcript contract (empty text is valid; missing text is an error)**

- [ ] **Design sentence-boundary detection heuristics**

  - Define safe speech boundaries (punctuation, pauses, length thresholds)
  - Avoid half-sentence or half-word speech output

- [ ] **Sketch incremental speech coordinator**

  - Lives in the state machine layer
  - Buffers streamed LLM output
  - Emits speakable chunks to TTS
  - Supports cancellation and barge-in

- [ ] **Measure LLM time-to-first-token vs full completion**

  - Log and compare latency improvements from streaming

- [~] **LAN OpenAI-compatible STT provider** (initial implementation)

- [x] **LAN OpenAI-compatible TTS provider** (initial implementation)

- [ ] **Allow selecting TTS audio format (pcm/mp3/opus) via config**

- [x] **Streaming TTS transport** (LAN HTTP provider)

- [x] **Streaming AudioStream decoding**

- [x] **MP3/OPUS incremental decoding**

- [x] **Streaming TTS playback**

- [x] **Application-owned system prompt (explicit, voice-first)**

- [ ] **Investigate/validate no-truncation playback for streaming TTS (ensure tail not cut off)**

- [ ] **Add completeness tests for ffmpeg decoding (tail samples non-zero)**

- [ ] **MP3/OPUS streaming decoding validated on Pi**

- [ ] **Playback latency measurements**

- [ ] **Optional jitter buffering / chunk sizing tuning**

- [ ] **Validate streaming TTS over real LAN (MP3/OPUS) with barge-in**

- [ ] **Drain ffmpeg stderr to avoid pipe backpressure**

- [ ] **Optional AudioStream.close() support across playback**

- [ ] **Document playback stop latency guarantees**

- [ ] **Measure time-to-first-audio (TTS streaming)**

- [ ] **Validate ffmpeg streaming behavior on Pi Zero**

- [ ] **Optional configurable decoder chunk size**

- [ ] **Optional selector timeout tuning for lower barge-in latency**

- [ ] **Allow non-PCM default for LAN TTS (MP3/OPUS first-byte streaming)**

- [ ] **Document AudioStream lifecycle expectations**

- [ ] **Add test coverage for streaming WAV header priming**

- [ ] **Add integration test for barge-in during streaming TTS**

- [ ] **Expose playback stop latency metrics**

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

- [ ] **Language-specific system prompts (explicitly deferred)**

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

## UX / Hardware (optional, non-blocking)

- [ ] Optional: richer LED patterns or audio-reactive effects (post-v1)

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

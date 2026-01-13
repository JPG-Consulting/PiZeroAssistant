# DEVELOPMENT

This document describes the Voice Assistant implementation and its behavioral contracts. It is aligned to the current code and makes no aspirational claims.

## Architecture overview

Voice Assistant is organized as a deterministic, event-driven pipeline with an always-on wakeword service:

- Audio capture thread produces fixed-size PCM frames.
- Wakeword service consumes frames for VAD and KWS, maintains a pre-roll buffer, and emits wake events.
- Main state machine reacts to wake events and advances through `IDLE → WAKE → RECORD → STT → LLM → TTS → IDLE`.
- Playback controller handles interruptible audio output.

### UX Event Contract (Authoritative)

- UX is event-driven and must not infer behavior from internal state.
- The `AssistantStateMachine` is the sole emitter of UX events.
- Events are semantic and may repeat.
- Current authoritative events include: `ASSISTANT_IDLE`, `ASSISTANT_READY`, `WAKE_DETECTED`, `LISTENING`, `PROCESSING`, `SPEAKING`, `NO_SPEECH`, `ERROR`.
- UX backends must render events only and must not implement logic or state transitions.
- Providers must never emit UX events.

## Audio pipeline

- **Capture**: `AudioCaptureThread` reads `int16` PCM frames at a fixed duration (default 20 ms). The capture callback never blocks; it drops frames when queues are full.
- **Frame sizes**: Frames are fixed-size; invalid frames are dropped before inference.
- **Decoupling**: Capture distributes frames to bounded queues for wakeword detection and recording. Inference never runs in the callback.

### Playback Stop Semantics

- Playback is interruptible via barge-in at all times.
- Barge-in is detected via wakeword events during playback.
- `PlaybackController` records stop reasons including: `normal_end`, `barge_in`, `explicit_stop`, `unknown`.
- Stop latency is bounded by decoder select timeout (~100 ms).
- Truncation detection is heuristic, based on expected vs actual duration, and logged as a warning.

### Wake Beep Behavior

- Wake beep playback is optional and configuration-driven.
- Failure to play the wake beep must never fail the pipeline.
- Wake beep playback is interruptible like TTS.
- Wake beep playback must not block recording or state transitions.
- Loudness must be baked into the audio asset; no runtime volume scaling is permitted.

## Wakeword service

- Dedicated thread (`WakewordService`) uses WebRTC VAD on 10/20/30 ms frames and openWakeWord on aggregated windows (default 80 ms).
- Frames are not shared between VAD and KWS paths; the service copies incoming frames before use.
- A fixed-size pre-roll buffer is maintained via a ring buffer (default 400 ms). Pre-roll is emitted as part of the wake event payload.
- Wakeword detection applies a cooldown to prevent repeated triggers.
- Wakeword inference input is `np.ndarray[int16]` mono PCM at 16 kHz.

### Wake-word model formats and selection

- Supported formats: `.onnx` and `.tflite`.
- The backend is selected by file extension (`.onnx` → ONNXRuntime, `.tflite` → TFLite).
- No auto-conversion, runtime downloading, or training logic exists in the runtime.
- A wake-word model must be present at startup, and the model format must match openWakeWord expectations.
- Model loading happens once at wakeword service initialization, not per frame.
- The wake-word service fails fast with a clear error when the model path is missing or misconfigured.

### Wakeword backend and model compatibility

The wakeword backend and model format must match exactly. This is a hard invariant enforced during initialization.
Backend selection is explicit and controlled by this project, and startup fails if the backend/model pairing is invalid.

Valid combinations:

- `tflite` backend → `.tflite` model
- `onnx` backend → `.onnx` model

openWakeWord supports multiple inference runtimes, but automatic backend detection is intentionally avoided.
Deterministic behavior is required on embedded systems, and silent fallback would hide performance and correctness issues.

If the invariant is violated, startup fails early with an explicit, user-actionable error, and the assistant never
enters the idle listening state. This is by design.

On Raspberry Pi Zero / Zero 2, TFLite is recommended for performance and footprint. ONNX is supported but heavier,
so the backend choice remains an explicit trade-off.

## Provider routing

- Providers are configured in ordered lists (`stt_providers`, `llm_providers`, `tts_providers`).
- Each provider has independent failure counters and cooldown windows.
- The router attempts providers in order and stops once `max_fallbacks` is exceeded.

## Provider architecture and extensibility

The provider system is an abstraction boundary. Each service type (STT, LLM, TTS) exposes a common provider interface, and the runtime only speaks to those interfaces. Providers are instantiated from configuration via a factory (`src/voiceassistant/providers/factory.py`) so new providers can be added without touching the state machine.

Provider-specific logic means anything beyond invoking the interface methods (for example, API payload formatting, endpoint routing, or response parsing). That logic should live inside provider implementations, not in the state machine, router, or audio pipeline.

### Endpoint Ownership and Third-Party API Assumption

- All external services (including services running on the local network) are treated as third-party APIs.
- Providers must not assume control over, or stability of, upstream API semantics.
- Configuration must specify the full HTTP endpoint for the specific operation performed by the provider.
- The `endpoint` field represents a complete, operation-specific URL.
- Providers must not compose, append, infer, or modify endpoint paths.
- Providers may adapt request and response wire formats, but must not alter endpoint structure.
- Any upstream API change must surface as an explicit failure rather than silent adaptation.
- Uniform treatment of LAN-hosted and internet-hosted services is intentional and required.

### Provider roles

- **STT providers** accept WAV audio and return transcribed text.
  STT providers may return an empty transcript (`""`) to indicate silence or no detected speech; this is not an error. Missing or malformed `text` fields are treated as provider failures.
  All STT providers must implement the STTProvider interface and expose exactly one operation: transcribe(wav_bytes) → STTResponse; the interface exists for typing and invariants only and must not contain shared behavior.
- **LLM providers** accept an explicit message list (`LLMRequest.messages`) plus an explicit system prompt (`LLMRequest.system_prompt`) and return response text.
  **LLM providers must not retain conversational history or dialogue state.** Each call must be treated as an independent completion; providers must not store prompts, responses, or message history across requests.
  - LLM providers may stream internally for latency improvements, but they still return a single `LLMResponse` text payload to the state machine.
  - Incremental speech and sentence boundary decisions belong to the state machine, not the provider.
- **TTS providers** accept text and return encoded audio. They may return either full WAV bytes or a streaming `AudioStream`.
  - All TTS providers must implement the TTSProvider interface and expose exactly one operation: synthesize(text) → TTSResponse; the interface exists for typing and invariants only and must not contain shared behavior.
  - LAN TTS uses `POST /v1/audio/speech` and requests `format: pcm` so the runtime receives raw audio bytes.
  - LAN TTS may return streaming audio or full payloads and must remain stateless like all other providers.
  - Playback owns decoding, buffering, and streaming control; providers must not decode audio or buffer entire output before playback begins.

### OpenAI Providers

- OpenAI LLM and TTS providers are treated as third-party APIs.
- Configuration must specify full, operation-specific endpoints for OpenAI providers.
- Providers must not construct or modify endpoint paths.
- OpenAI LLM providers use the Chat Completions API and are fully stateless.
- OpenAI TTS providers use the Audio → Speech API and may stream audio.
- OpenAI TTS providers may return different encoded formats based on Content-Type.
- Audio decoding, buffering, and playback control remain owned by the runtime.
- API keys are supplied via environment variables only.

`provider_type: local_echo` is a diagnostic-only LLM provider that returns the latest user message verbatim. It has no intelligence, memory, context, or external dependency, and exists solely to validate the STT → LLM → TTS pipeline. It is not a conversational model.

All providers are synchronous and must raise `ProviderError` on failure. “Stateless” means providers must not retain cross-request conversational, audio, or session state (no carried-over context, buffers, or history). Providers may still read configuration, keep internal helpers, and use per-call transient state.

**Conversational memory placement:** Conversational memory is owned by the state machine and supplied explicitly with each LLM request. Providers remain unaware of dialogue continuity and must never implement provider-side chat history. Persistence, when enabled, is local-only and opt-in.

### Conversation Memory Lifecycle

- Conversation memory is owned exclusively by the state machine.
- Memory reset commands are evaluated after STT and before LLM.
- Empty STT transcripts do not mutate memory.
- Assistant replies are added to memory only after successful LLM completion.
- Persistence, when enabled, occurs only after a successful assistant reply.

### LLM token limits (Phase 1)

- The application MAY configure a per-request maximum token limit for LLM providers.
- Token limits are a routing/request policy, not a provider responsibility.
- Providers remain stateless and must not track usage.
- If a provider supports a native `max_tokens` (or equivalent) parameter, it is used.
- If a provider does not support native limits, Phase 1 does not enforce bounds on responses.
- Absence of a configured limit means unbounded behavior (current default).
- Rolling budgets, time windows, and usage accounting are explicitly out of scope for Phase 1.

#### Future design: Phase 2 token budgeting (not implemented)

- Phase 2 introduces per-provider token budgets (hour/day).
- Enforcement remains application-owned (router-level), not provider-owned.
- Providers remain stateless and unaware of budgets.
- Token accounting may use provider-reported usage when available, with deterministic estimation as a fallback.
- Providers exceeding budget enter a cooldown state, similar to failure-based cooldowns.
- Optional user-facing UX feedback (spoken warnings) may be added later.
- Phase 2 is intentionally deferred to avoid complexity in Phase 1.

### System prompt ownership

- The system prompt is application-owned and always passed explicitly with every LLM call.
- The prompt is loaded from `assets/prompts/system.md`.
- If the asset is missing or empty, a built-in default from `src/voiceassistant/llm/prompts.py` is used.
- The assistant is voice-first; all responses are optimized for text-to-speech playback.

### Provider capabilities (audio formats)

- STT providers declare supported input formats via `audio_formats` (for example, `["wav", "opus"]`).
- `wav` is the required baseline and must always be listed.
- Unsupported `audio_formats` values fail fast during config load (no runtime fallback or silent correction).
- These flags are declarative metadata only and do not imply runtime encoding support or negotiation; providers do not negotiate formats and the runtime does not transcode.
- The pipeline decides which encoding to send. Today, it only sends WAV regardless of the declared list.

### Configuration-driven selection

Providers are selected via `config.yaml`. Multiple providers can be listed per service and are tried in order for fallback. Provider names are logical identifiers only (no name-prefix selection). The provider implementation is selected explicitly by `provider_type`, and unsupported types fail fast at startup during config load.

Example:

```yaml
routing:
  stt_providers:
    - name: "lan-stt"
      provider_type: "http"
      endpoint: "http://<host>/v1/audio/transcriptions"
      timeout_s: 15
      api_key_env: null
      max_failures: 2
      cooldown_s: 60
      audio_formats: ["wav"]
  llm_providers:
    - name: "local-llm"
      provider_type: "http"
      endpoint: "http://<host>/v1/chat/completions"
      timeout_s: 20
      api_key_env: null
      max_failures: 2
      cooldown_s: 60
  tts_providers:
    - name: "local-tts"
      provider_type: "http"
      endpoint: "http://<host>/v1/audio/speech"
      timeout_s: 20
      api_key_env: null
      max_failures: 2
      cooldown_s: 60
  max_fallbacks: 2
```

### Where provider code lives

- Provider implementations live in `src/voiceassistant/providers/`.
- Routing and fallback live in `src/voiceassistant/providers/router.py` and should not be modified for new providers.
- Base types live in `src/voiceassistant/providers/base.py` (including `Provider` and `ProviderError`).

### Adding a new provider

**To add a new provider:**

1. Implement a new provider class (for example, `GoogleSTTProvider`).
2. Subclass the appropriate base provider.
3. Implement the required method (`transcribe`, `complete`, or `synthesize`).
4. Ensure configuration fields are read from `ProviderConfig`.
5. Register the new provider type in `src/voiceassistant/providers/factory.py`.

Existing providers should not be edited; add new providers as additive implementations.

#### Service-specific checklist

- **STT**: implement `transcribe(wav_bytes: bytes) -> STTResponse`, accept WAV bytes, return text.
- **LLM**: implement `complete(req: LLMRequest) -> LLMResponse`, accept explicit messages, return response text.
- **TTS**: implement `synthesize(text: str) -> TTSResponse`, accept text, return WAV bytes.

Invariants for all providers:

- Raise `ProviderError` for any failure.
- Respect `timeout_s` from configuration.
- Do not retry internally (the router handles fallback).

### Non-goals

- No dynamic plugin loading (yet).
- No runtime code generation.
- No provider auto-detection.
- No provider-specific logic in the state machine.

## Health checks

- Providers can opt into health checks via `Provider.health_check()`; default is always healthy.
- Failures increment per-provider counters; after `max_failures`, the provider enters cooldown for `cooldown_s` seconds.

## Fallback behavior

- Each pipeline stage (STT, LLM, TTS) uses its own router instance.
- When a provider fails, the router selects the next eligible provider.
- If all providers fail, the pipeline aborts and returns to `IDLE`.

## Timeout semantics

- All HTTP requests use explicit per-provider timeouts defined in configuration.
- Recorder waits for at most `max_record_seconds + 1` seconds before aborting the recording step.

## Error-handling guarantees

- Pipeline errors are contained to the current run; exceptions are logged and the assistant returns to `IDLE`.
- Wakeword detection and audio capture are not stopped when a pipeline fails.

## Recovery semantics

- On any error after a wake event, the state machine resets to `IDLE`.
- Playback is stopped immediately when barge-in is detected.

### Error Handling and Recovery Paths

- Any provider failure aborts the current pipeline run.
- On failure, the assistant always returns to `IDLE`.
- Wakeword detection and audio capture are never stopped by pipeline failures.
- Partial pipeline progress must not mutate conversation memory unless explicitly completed.

## Behavioral invariants

- Audio capture never blocks on inference.
- Wakeword detection remains active at all times.
- Wakeword events are delivered via a bounded queue.
- Recording output is always written to `/tmp/last_command.wav` unless configured otherwise.

## Performance invariants

- Bounded queues protect against unbounded memory growth.
- Frame drops are logged at shutdown for visibility.

## Privacy guarantees & invariants

- Audio is processed locally until a wakeword event triggers recording.
- Network calls occur only after a wakeword-triggered recording is complete.
- No telemetry is emitted.

### Recording Artifact Guarantees

- The last user command recording is written to a fixed path (default `/tmp/last_command.wav`).
- The file may contain silence or partial speech.
- The file is overwritten on each command.
- The artifact is local-only and considered sensitive data.
- Presence of this file does not imply successful STT or LLM execution.

## Power-management & real-time constraints

- The capture callback performs only byte copying and non-blocking queue operations.
- Inference and network calls run in dedicated threads outside of capture.
- Barge-in uses wakeword detection during playback and stops audio output immediately.

## Configuration invariants

- YAML configuration is required; missing required fields raise `ConfigError`.
- Secrets are read from environment variables defined in `api_key_env`.

## Known failure modes

- Missing `openwakeword` dependency prevents wakeword service from starting.
- Invalid WAV bytes from a TTS provider result in playback errors.
- If provider endpoints return malformed JSON, the pipeline aborts and returns to `IDLE`.
- Python 3.11 PGO/LTO builds fail on Pi Zero / Zero 2; this is an environmental limitation, not a bug. See `docs/dev/python-versions.md` for details.

## Logging contract (v1.0)

- All modules use `logging.getLogger(__name__)`.
- Log messages are stable and suitable for monitoring.
- No per-frame logging is performed.

### Python version requirements

Python 3.11 is required because `openwakeword==0.5.1` only supports Python 3.11 today. Raspberry Pi OS Bookworm ships with Python 3.11, while Raspberry Pi OS Trixie ships with Python 3.13 and requires a separate Python 3.11 install. See `docs/dev/python-versions.md` for platform details and installation guidance.

Building Python 3.11 on Pi Zero / Zero 2 has known constraints, and PGO/LTO builds are explicitly unsupported. The procedures and limitations described in `docs/dev/python-versions.md` are part of the supported platform contract for this project.

## Platform support policy

This project intentionally targets a narrow, explicitly supported set of platforms. The primary supported environment is Raspberry Pi Zero / Zero 2 running Raspberry Pi OS (Bookworm or Trixie). Python 3.11 is an intentional, documented, non-negotiable requirement for v1.0, and dependency compatibility (for example, openWakeWord) can dictate the supported platform and Python requirements. See `docs/dev/python-versions.md` for the authoritative platform constraints.

Support is intentional, not universal. Other platforms may work, but they are not guaranteed. Platform changes are handled deliberately and documented when support is added or removed.

Non-goals:

- Support for all Python versions.
- Support for all Linux distributions.
- Support for arbitrary hardware.
- Backward compatibility with deprecated platforms.

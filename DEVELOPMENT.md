# DEVELOPMENT

This document describes the Voice Assistant implementation and its behavioral contracts. It is aligned to the current code and makes no aspirational claims.

## Architecture overview

Voice Assistant is organized as a deterministic, event-driven pipeline with an always-on wakeword service:

- Audio capture thread produces fixed-size PCM frames.
- Wakeword service consumes frames for VAD and KWS, maintains a pre-roll buffer, and emits wake events.
- Main state machine reacts to wake events and advances through `IDLE → WAKE → RECORD → STT → LLM → TTS → IDLE`.
- Playback controller handles interruptible audio output.

## Audio pipeline

- **Capture**: `AudioCaptureThread` reads `int16` PCM frames at a fixed duration (default 20 ms). The capture callback never blocks; it drops frames when queues are full.
- **Frame sizes**: Frames are fixed-size; invalid frames are dropped before inference.
- **Decoupling**: Capture distributes frames to bounded queues for wakeword detection and recording. Inference never runs in the callback.

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

### Provider roles

- **STT providers** accept WAV audio and return transcribed text.
- **LLM providers** accept an explicit message list (`LLMRequest.messages`) and return response text.
  **LLM providers must not retain conversational history or dialogue state.** Each call must be treated as an independent completion; providers must not store prompts, responses, or message history across requests.
- **TTS providers** accept text and return WAV audio.

All providers are synchronous and must raise `ProviderError` on failure. “Stateless” means providers must not retain cross-request conversational, audio, or session state (no carried-over context, buffers, or history). Providers may still read configuration, keep internal helpers, and use per-call transient state.

**Conversational memory placement:** Conversational memory is owned by the state machine and supplied explicitly with each LLM request. Providers remain unaware of dialogue continuity and must never implement provider-side chat history. Persistence, when enabled, is local-only and opt-in.

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

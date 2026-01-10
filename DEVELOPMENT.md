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

### Wake-word model formats and selection

- Supported formats: `.onnx` and `.tflite`.
- The backend is selected by file extension (`.onnx` → ONNXRuntime, `.tflite` → TFLite).
- No auto-conversion, runtime downloading, or training logic exists in the runtime.
- A wake-word model must be present at startup, and the model format must match openWakeWord expectations.
- Model loading happens once at wakeword service initialization, not per frame.
- The wake-word service fails fast with a clear error when the model path is missing or misconfigured.

## Provider routing

- Providers are configured in ordered lists (`stt_providers`, `llm_providers`, `tts_providers`).
- Each provider has independent failure counters and cooldown windows.
- The router attempts providers in order and stops once `max_fallbacks` is exceeded.

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

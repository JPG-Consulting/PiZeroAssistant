# Wake-Word Detection Pipeline

This document describes the **wake-word detection pipeline** used in this project.
It defines the end-to-end flow from audio capture to wake decision, along with the
key invariants and design constraints that must be preserved.

This is a **descriptive specification**, not a tutorial and not an implementation guide.

---

## Scope and Goals

The wake-word pipeline is responsible for:
- Detecting the presence of a specific wake word in continuous audio
- Operating fully offline
- Running reliably on constrained hardware
- Producing stable behavior across environments and time

The pipeline is designed to be:
- Simple
- Deterministic
- Aligned between training and runtime

---

## High-Level Pipeline Stages

The wake-word system consists of the following stages:

1. Audio capture  
2. Framing and buffering  
3. Feature extraction  
4. Model inference  
5. Decision logic  

Each stage has clearly defined inputs, outputs, and invariants.

---

## 1. Audio Capture

**Input**
- Mono PCM16 audio
- Fixed sample rate (e.g. 16 kHz)
- Captured through the same microphone path used in deployment

**Invariants**
- Audio is not normalized, rescaled, or dynamically amplified
- Microphone gain and hardware characteristics are preserved
- No resampling occurs at runtime

The wake-word system relies on the raw acoustic characteristics of the deployment environment.
Recording remains raw int16 PCM with no software gain or limiter applied.

---

## 2. Framing and Buffering

**Purpose**
- Convert the continuous audio stream into fixed-duration analysis windows

**Behavior**
- Audio is buffered into sliding windows of fixed length
- Window duration matches the training clip duration (`clip-sec`)
- Windows are evaluated independently

**Invariants**
- Window length is fixed and consistent across training and runtime
- No attempt is made to segment or align the wake word within the window

---

## 3. Feature Extraction

**Purpose**
- Convert raw audio into model-ready features

**Behavior**
- Before feature extraction, a fixed, deterministic runtime preprocessing step applies a
  non-adaptive software gain and a fixed soft limiter
- A deterministic feature extractor (e.g. log-mel spectrogram) is then applied
- Feature parameters are fixed and explicitly configured

**Invariants**
- Runtime preprocessing is configuration- and code-defined, not adaptive normalization or AGC
- Training preprocessing mirrors runtime preprocessing exactly
- Feature extraction used in training and runtime must be identical
- No feature normalization or augmentation is applied at runtime
- Feature shapes and scaling are stable and configuration-driven

---

## 4. Model Inference

**Purpose**
- Estimate the probability that a window contains the wake word

**Behavior**
- The model processes one feature window at a time
- The output is a scalar wake probability

**Invariants**
- The model produces probabilities, not decisions
- The model sees exactly the same type of input at training and runtime
- No temporal context is assumed beyond the current window

---

## 5. Decision Logic

**Purpose**
- Convert per-window probabilities into wake events

**Behavior**
- Thresholding and optional temporal smoothing may be applied
- Decision logic operates outside the model

**Invariants**
- Policy decisions (thresholds, debounce, cooldown) are separated from inference
- The model is never retrained to compensate for policy choices

---

## Configuration Mapping (Conceptual)

The wake-word pipeline is configured via `config.yaml`.
Configuration values are grouped by **which pipeline stage they affect**, not by implementation details.

This mapping is **conceptual** and intended to clarify ownership and invariants.

### Audio Capture
Configuration keys and responsibilities:
- `audio.sample_rate`: defines the fixed capture sample rate
- `audio.device`: selects the capture device identity
- `audio.channels`: defines the channel count
- `audio.dtype`: defines the raw sample format
- `audio.block_ms`: sets the capture block timing granularity

These settings define the raw audio format and must remain consistent between training and runtime.

---

### Framing and Buffering
Configuration keys and responsibilities:
- `features.clip_seconds`: defines the analysis window duration
- `features.hop_ms`: defines the analysis window stride timing
- `audio.block_ms`: defines the buffering cadence used to form windows

These values define *when* inference happens, not *what* is inferred.

---

### Feature Extraction
Configuration keys and responsibilities:
- `features.n_fft`: defines FFT size
- `features.win_ms`: defines analysis window length
- `features.hop_ms`: defines feature frame hop
- `features.n_mels`: defines mel band count
- `features.fmin`: defines lower frequency bound
- `features.fmax`: defines upper frequency bound
- `features.log_eps`: defines log scaling floor

All feature-related parameters must match exactly between:
- offline dataset inspection
- model training
- runtime inference

Any change in feature configuration constitutes a **pipeline change** and requires retraining.

---

### Model Inference
Configuration keys and responsibilities:
- `wakeword.type`: selects the inference backend
- `wakeword.model_path`: identifies the model artifact
- `wakeword.input_name`: defines input tensor naming expectations
- `wakeword.output_name`: defines output tensor naming expectations

The model is treated as a black box with a stable input/output contract.

---

### Decision Logic
Configuration keys and responsibilities:
- `wakeword.threshold`: defines the wake probability threshold
- `wakeword.consecutive_hits`: defines the consecutive-hit requirement
- `wakeword.cooldown_ms`: defines the post-trigger cooldown timing

These settings affect user-facing behavior but must not influence training data or feature extraction.

---

## Dataset Preparation and Its Role

Dataset preparation determines **which audio examples exist**, not how they are modified.

Key principles:
- Audio is never normalized, rescaled, or augmented
- Wake-word clips are handled conservatively (one centered clip per recording)
- Negative clips are curated for realism and diversity
- Training data must reflect runtime conditions as closely as possible

Dataset curation is intentionally conservative to preserve semantic correctness.

---

For audio-parity invariants and the authoritative preprocessing reference, see
`docs/audio_parity_checklist.md`.

## When Retraining Is Required

This checklist is a decision aid for determining whether a change requires retraining.

**Retraining is required when:**
- Feature extraction configuration changes (mel parameters, FFT size, frequency range)
- Analysis window duration (`clip_seconds`) changes
- Training data distribution shifts materially (new microphone, new acoustic domain)
- The model architecture or output contract changes

**Retraining is not required when:**
- Decision logic parameters change (thresholds, debounce, cooldown)
- Runtime-only policy behavior changes
- Logging, metrics, or evaluation tooling changes

---

## Common Failure Modes

These are system-level failure modes that arise when pipeline invariants are violated.

- Feature mismatch between training and runtime (cause) leads to unstable or incorrect detections (symptom)
- Overrepresentation of trivial negatives like silence (cause) yields poor real-world discrimination (symptom)
- Dataset distribution drift (cause) reduces accuracy in deployment environments (symptom)
- Over-tuning thresholds to compensate for poor data (cause) causes brittle wake behavior (symptom)
- Treating wake-word detection as a policy problem instead of a modeling problem (cause) leads to inconsistent decisions (symptom)

---

## Explicit Non-Goals

The wake-word pipeline does **not** include:
- Online data augmentation
- Adaptive gain control
- Wake-word segmentation or alignment
- Language understanding or intent classification
- Automatic rejection of wake words due to clipping or loudness

Introducing any of the above constitutes a design change and must be evaluated explicitly.

---

## Design Philosophy

The wake-word pipeline prioritizes:
- Stability over cleverness
- Data quality over data volume
- Explicit contracts over implicit assumptions

Changes to any stage should be evaluated in terms of their impact on the **entire pipeline**, not in isolation.

---

## Summary

The wake-word system is a small, well-defined pipeline with strict invariants.
Its reliability depends on maintaining alignment between data, features, model, and runtime.

This document exists to make those assumptions explicit.

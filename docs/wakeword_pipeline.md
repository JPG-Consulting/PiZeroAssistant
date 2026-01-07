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
- A deterministic feature extractor (e.g. log-mel spectrogram) is applied
- Feature parameters are fixed and explicitly configured

**Invariants**
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
Typical configuration responsibilities:
- Sample rate
- Audio device selection
- Channel configuration

These settings define the raw audio format and must remain consistent between training and runtime.

---

### Framing and Buffering
Typical configuration responsibilities:
- Analysis window duration (`clip-sec`)
- Buffering or hop behavior

These values define *when* inference happens, not *what* is inferred.

---

### Feature Extraction
Typical configuration responsibilities:
- FFT size and hop length
- Number of mel bands
- Frequency range
- Log scaling behavior

All feature-related parameters must match exactly between:
- offline dataset inspection
- model training
- runtime inference

Any change in feature configuration constitutes a **pipeline change** and requires retraining.

---

### Model Inference
Typical configuration responsibilities:
- Model path or identifier
- Input/output tensor expectations

The model is treated as a black box with a stable input/output contract.

---

### Decision Logic
Typical configuration responsibilities:
- Wake probability threshold
- Debounce or cooldown timing
- Optional temporal smoothing parameters

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

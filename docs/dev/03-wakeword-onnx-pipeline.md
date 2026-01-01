# Wake-word ONNX Pipeline (Developer Notes)

## Overview
This project uses a wake-word detector running fully offline on Raspberry Pi Zero 2 W.
Wake-word models are trained externally and deployed as ONNX files.

This document describes:
- the ONNX input/output contract
- the role of the dummy wake-word model
- how inference is wired in the assistant

This is **developer documentation**. End users are not expected to train or modify
wake-word models directly on the Raspberry Pi.

---

## Dummy Wake-word Model (Smoke Test)

A dummy ONNX model (`dummy_wake.onnx`) is used during development to validate:

- ONNXRuntime availability on ARM (Raspberry Pi Zero 2 W)
- feature extraction correctness
- wake-word detector logic (thresholds, consecutive hits, cooldown)
- metrics and observability

This model is **NOT** a real wake-word detector.

It always outputs a fixed wake probability and is only intended for
end-to-end infrastructure testing.

### Why it exists
- isolate infrastructure and wiring bugs from model quality
- enable early testing before a real wake-word model is trained
- provide a deterministic signal for debugging

### Why it is not versioned
- models are binary artifacts and can be large
- real wake-word models are project- or user-specific
- the repository defines the **interface**, not the trained weights

The `models/` directory is intentionally excluded from version control.

---

## ONNX Model Contract

All wake-word ONNX models used by this project **must** follow this contract.

### Input

- **Name:** `logmel` (configurable via YAML)
- **Shape:** `(1, n_mels, frames)`
- **Dtype:** `float32`
- **Content:** log-mel spectrogram extracted from raw audio

Notes:
- Batch size is always `1`
- `n_mels` and `frames` must match the feature extraction configuration

### Output

Supported output formats:

- `(1,)` → wake probability (`p_wake`)
- `(1, 2)` → `[not_wake, wake]` scores or logits

The detector normalizes both formats internally and always works with a single
wake probability value.

---

## Model Generation Notes

ONNX models **must** be generated with versions compatible with ARM builds of
ONNXRuntime.

Required constraints:

- **IR version:** ≤ 11 (recommended: 7)
- **Opset version:** ≤ 11

Models generated with newer IR or opset versions may fail to load on
Raspberry Pi devices.

---

## Dummy Wake-word Model Generation (Reproducibility)

The dummy wake-word model used for smoke testing can be generated with a small,
standalone Python script.

This script can be executed on:
- a local PC
- Google Colab
- any environment with Python and ONNX tooling

### Dependencies

```bash
pip install onnx numpy
```

### Generation Script

```python
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nh
import numpy as np

N_MELS = 40
T = 98  # ~1 second with 10 ms hop

# Input tensor
inp = oh.make_tensor_value_info(
    "logmel",
    onnx.TensorProto.FLOAT,
    [1, N_MELS, T],
)

# Output tensor
out = oh.make_tensor_value_info(
    "prob",
    onnx.TensorProto.FLOAT,
    [1],
)

# Constant wake probability
const_tensor = nh.from_array(
    np.array([0.95], dtype=np.float32),
    name="const_prob",
)

const_node = oh.make_node(
    "Constant",
    inputs=[],
    outputs=["prob"],
    value=const_tensor,
)

graph = oh.make_graph(
    [const_node],
    "dummy_wake",
    [inp],
    [out],
)

# Explicit opset (ARM compatible)
opset = oh.make_operatorsetid("", 11)

model = oh.make_model(
    graph,
    opset_imports=[opset],
    producer_name="dummy_wake",
)

# Explicit IR version (ARM compatible)
model.ir_version = 7

onnx.checker.check_model(model)
onnx.save(model, "dummy_wake.onnx")
```

The resulting `dummy_wake.onnx` file can then be copied to the `models/`
directory on the Raspberry Pi for ONNX smoke testing.

---

## Where Wake-word Training Happens

Wake-word training is performed **offline** on a PC or in Google Colab.
The Raspberry Pi only performs **inference**.

A separate document will describe the wake-word training pipeline once it is
finalized and stabilized.

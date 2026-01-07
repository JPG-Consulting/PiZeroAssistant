# TODO — PiZeroAssistant

## Short-term
- [ ] Decide whether to keep legacy opset 11 dummy model as-is or move to tools/legacy
- [ ] Run gold_test_onnx.py with newly trained opset 18 model on Pi

## Medium-term
- [ ] Add CI check for hardcoded log-mel frame counts
- [ ] Evaluate ONNXRuntime INT8 quantization

## Long-term
- [ ] Wake-word metrics collection (false positives over time)
- [ ] Multi-language wake-word support (if needed)

import onnx
import onnx.helper as oh
import onnx.numpy_helper as nh
import numpy as np

N_MELS = 40
T = 98  # ~1s with 10ms hop

# Input
inp = oh.make_tensor_value_info(
    "logmel",
    onnx.TensorProto.FLOAT,
    [1, N_MELS, T],
)

# Output
out = oh.make_tensor_value_info(
    "prob",
    onnx.TensorProto.FLOAT,
    [1],
)

# Constant output node
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
    nodes=[const_node],
    name="dummy_wake",
    inputs=[inp],
    outputs=[out],
)

# Explicit opset (VERY IMPORTANT)
opset = oh.make_operatorsetid("", 11)

model = oh.make_model(
    graph,
    opset_imports=[opset],
    producer_name="dummy_wake",
)

# Explicit IR version (VERY IMPORTANT)
model.ir_version = 7

onnx.checker.check_model(model)
onnx.save(model, "dummy_wake.onnx")

print("Saved dummy_wake.onnx (IR v7, opset 11)")

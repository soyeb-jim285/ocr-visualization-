"""Extract conv1 weights from the ONNX model to a JSON file."""

import json
import onnx
import numpy as np
from onnx import numpy_helper

model = onnx.load("public/models/emnist-cnn/model.onnx")

weights = None
biases = None

for init in model.graph.initializer:
    name = init.name.lower()
    if "conv1" in name or "0.weight" in name:
        if "bias" not in name:
            w = numpy_helper.to_array(init)
            if w.shape == (32, 1, 3, 3):
                weights = w
    if "conv1" in name or "0.bias" in name:
        if "bias" in name:
            b = numpy_helper.to_array(init)
            if b.shape == (32,):
                biases = b

if weights is None or biases is None:
    # Fallback: iterate all initializers and pick by shape
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.shape == (32, 1, 3, 3) and weights is None:
            weights = arr
            print(f"Found weights: {init.name}")
        elif arr.shape == (32,) and biases is None:
            biases = arr
            print(f"Found biases: {init.name}")

assert weights is not None, "Could not find conv1 weights"
assert biases is not None, "Could not find conv1 biases"

print(f"Weights shape: {weights.shape}, range: [{weights.min():.4f}, {weights.max():.4f}]")
print(f"Biases shape: {biases.shape}, range: [{biases.min():.4f}, {biases.max():.4f}]")

data = {
    "weights": [round(float(x), 6) for x in weights.flatten()],
    "biases": [round(float(x), 6) for x in biases.flatten()],
    "shape": [32, 1, 3, 3],
}

with open("public/models/emnist-cnn/conv1-weights.json", "w") as f:
    json.dump(data, f)

print(f"Saved to public/models/emnist-cnn/conv1-weights.json ({len(json.dumps(data))} bytes)")

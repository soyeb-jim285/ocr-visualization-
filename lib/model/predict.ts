import * as ort from "onnxruntime-web";
import { loadModel } from "./loadModel";
import { BYMERGE_MERGED_INDICES } from "./classes";
import type { LayerActivations } from "@/stores/inferenceStore";

/** Names of the intermediate outputs from the ONNX model (must match export) */
const INTERMEDIATE_OUTPUTS = [
  "conv1",
  "relu1",
  "conv2",
  "relu2",
  "pool1",
  "conv3",
  "relu3",
  "pool2",
  "dense1",
  "relu4",
  "output",
] as const;

/** Layer output shapes (NCHW for conv, N*units for dense) */
const LAYER_SHAPES: Record<string, { type: "conv"; c: number; h: number; w: number } | { type: "dense"; units: number }> = {
  conv1: { type: "conv", c: 32, h: 28, w: 28 },
  relu1: { type: "conv", c: 32, h: 28, w: 28 },
  conv2: { type: "conv", c: 64, h: 28, w: 28 },
  relu2: { type: "conv", c: 64, h: 28, w: 28 },
  pool1: { type: "conv", c: 64, h: 14, w: 14 },
  conv3: { type: "conv", c: 128, h: 14, w: 14 },
  relu3: { type: "conv", c: 128, h: 14, w: 14 },
  pool2: { type: "conv", c: 128, h: 7, w: 7 },
  dense1: { type: "dense", units: 256 },
  relu4: { type: "dense", units: 256 },
  output: { type: "dense", units: 62 },
};

export interface InferenceResult {
  prediction: number[];
  layerActivations: LayerActivations;
}

/**
 * Convert a flat Float32Array from ONNX (NCHW) to [C][H][W] for visualization.
 * Also transposes each channel (swap H/W) so the display orientation matches
 * how the user drew the character. The model operates on transposed EMNIST
 * images, so we transpose back for human-readable visualization.
 */
function nchwToChannels(
  data: Float32Array,
  c: number,
  h: number,
  w: number
): number[][][] {
  const channels: number[][][] = [];
  for (let ci = 0; ci < c; ci++) {
    const channel: number[][] = [];
    for (let hi = 0; hi < h; hi++) {
      const row: number[] = [];
      for (let wi = 0; wi < w; wi++) {
        // Transpose: read [wi][hi] instead of [hi][wi]
        row.push(data[ci * h * w + wi * w + hi]);
      }
      channel.push(row);
    }
    channels.push(channel);
  }
  return channels;
}

/**
 * Softmax over a Float32Array, masking out untrained ByMerge indices.
 * Merged class outputs are set to -Infinity before softmax so they get ~0 probability.
 */
function softmax(arr: Float32Array): number[] {
  // Mask untrained neurons
  const masked = Array.from(arr);
  for (const idx of BYMERGE_MERGED_INDICES) {
    masked[idx] = -Infinity;
  }
  const max = Math.max(...masked);
  const exps = masked.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/**
 * Run inference and extract all intermediate layer activations.
 *
 * Input: Float32Array of shape [1, 1, 28, 28] (NCHW)
 */
export async function runInference(
  inputData: Float32Array
): Promise<InferenceResult> {
  const session = await loadModel();

  // Create input tensor (NCHW: batch=1, channels=1, height=28, width=28)
  const inputTensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);

  // Run inference
  const results = await session.run({ input: inputTensor });

  const layerActivations: LayerActivations = {};

  for (const name of INTERMEDIATE_OUTPUTS) {
    const tensor = results[name];
    if (!tensor) continue;

    const data = tensor.data as Float32Array;
    const shape = LAYER_SHAPES[name];

    if (shape.type === "conv") {
      // Convert NCHW flat array to [C][H][W]
      layerActivations[name] = nchwToChannels(data, shape.c, shape.h, shape.w);
    } else {
      // Dense: just convert to number[]
      layerActivations[name] = Array.from(data);
    }
  }

  // The output layer needs softmax (PyTorch model outputs raw logits in multi-output mode)
  const outputData = results["output"]?.data as Float32Array;
  const prediction = outputData ? softmax(outputData) : [];

  return { prediction, layerActivations };
}

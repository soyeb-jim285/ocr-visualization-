import * as ort from "onnxruntime-web";
import { loadModel, invalidateSession } from "./loadModel";
import {
  INTERMEDIATE_OUTPUTS,
  LAYER_SHAPES,
  nchwToChannels,
  softmax,
} from "./modelUtils";
import type { LayerActivations } from "@/stores/inferenceStore";

export interface InferenceResult {
  prediction: number[];
  layerActivations: LayerActivations;
}

/**
 * Run inference and extract all intermediate layer activations.
 *
 * Input: Float32Array of shape [1, 1, 28, 28] (NCHW)
 */
export async function runInference(
  inputData: Float32Array
): Promise<InferenceResult> {
  let session = await loadModel();

  // Create input tensor (NCHW: batch=1, channels=1, height=28, width=28)
  const inputTensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);

  let results;
  try {
    results = await session.run({ input: inputTensor });
  } catch (e) {
    // Session may be corrupted — recreate it and retry once
    console.warn("ONNX session.run failed, recreating session:", e);
    invalidateSession();
    session = await loadModel();
    results = await session.run({
      input: new ort.Tensor("float32", inputData, [1, 1, 28, 28]),
    });
  }

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

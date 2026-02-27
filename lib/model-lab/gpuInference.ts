/**
 * Inference on a GPU-trained ONNX model using ONNX Runtime Web.
 * Extracts intermediate layer activations in the same format as customInference.ts.
 */

import * as ort from "onnxruntime-web";
import type { LayerActivations } from "@/stores/inferenceStore";

export interface GpuInferenceResult {
  prediction: number[];
  layerActivations: LayerActivations;
}

/**
 * Run inference on a GPU-trained ONNX model.
 *
 * @param session - ONNX InferenceSession from the GPU-trained model
 * @param intermediateLayerNames - Output names excluding "output" (from the training response)
 * @param inputData - Float32Array [1,1,28,28] NCHW (from preprocessCanvas)
 */
export async function runGpuModelInference(
  session: ort.InferenceSession,
  intermediateLayerNames: string[],
  inputData: Float32Array,
): Promise<GpuInferenceResult> {
  const inputTensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);
  const results = await session.run({ input: inputTensor });

  const layerActivations: LayerActivations = {};

  for (const name of intermediateLayerNames) {
    if (name === "output") continue;
    const tensor = results[name];
    if (!tensor) continue;

    const data = tensor.data as Float32Array;
    const shape = tensor.dims;

    if (shape.length === 4) {
      // NCHW: [1, C, H, W] → number[C][H][W]
      const [, c, h, w] = shape;
      const channels: number[][][] = [];
      for (let ci = 0; ci < c; ci++) {
        const channel: number[][] = [];
        for (let hi = 0; hi < h; hi++) {
          const row: number[] = [];
          for (let wi = 0; wi < w; wi++) {
            row.push(data[ci * h * w + hi * w + wi]);
          }
          channel.push(row);
        }
        channels.push(channel);
      }
      layerActivations[name] = channels;
    } else if (shape.length === 2) {
      // Dense: [1, units] → number[]
      layerActivations[name] = Array.from(data);
    }
  }

  // Output: apply softmax to raw logits
  const outputTensor = results["output"];
  let prediction: number[] = [];
  if (outputTensor) {
    const logits = outputTensor.data as Float32Array;
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) max = logits[i];
    }
    let sum = 0;
    prediction = new Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      prediction[i] = Math.exp(logits[i] - max);
      sum += prediction[i];
    }
    for (let i = 0; i < logits.length; i++) {
      prediction[i] /= sum;
    }
  }

  return { prediction, layerActivations };
}

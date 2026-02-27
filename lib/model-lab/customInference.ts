import type * as tf from "@tensorflow/tfjs";
import type { LayerActivations } from "@/stores/inferenceStore";

export interface CustomInferenceResult {
  prediction: number[];
  layerActivations: LayerActivations;
}

/**
 * Run inference on a custom TF.js model, extracting intermediate activations.
 *
 * Input: Float32Array [1, 1, 28, 28] (NCHW from preprocessCanvas).
 * For single-channel 28×28, NCHW and NHWC have identical memory layout,
 * so we just reshape to [1, 28, 28, 1].
 */
export function runCustomInference(
  tf: typeof import("@tensorflow/tfjs"),
  model: tf.LayersModel,
  intermediateLayerNames: string[],
  inputData: Float32Array,
): CustomInferenceResult {
  // Reshape NCHW [1,1,28,28] → NHWC [1,28,28,1] (same memory for single-channel)
  const inputTensor = tf.tensor4d(inputData, [1, 28, 28, 1]);

  // Build activation extraction model
  const intermediateOutputs = intermediateLayerNames
    .map((name) => {
      try {
        return model.getLayer(name).output;
      } catch {
        return null;
      }
    })
    .filter((o): o is tf.SymbolicTensor => o !== null);

  const finalOutput = model.output as tf.SymbolicTensor;
  const extractionModel = tf.model({
    inputs: model.input,
    outputs: [...intermediateOutputs, finalOutput],
  });

  const outputs = extractionModel.predict(inputTensor) as tf.Tensor[];
  inputTensor.dispose();

  const layerActivations: LayerActivations = {};
  const validNames = intermediateLayerNames.filter((name) => {
    try {
      model.getLayer(name);
      return true;
    } catch {
      return false;
    }
  });

  for (let i = 0; i < validNames.length; i++) {
    const tensor = outputs[i];
    const shape = tensor.shape;
    const data = tensor.dataSync() as Float32Array;

    if (shape.length === 4) {
      // NHWC: [batch, h, w, c] → convert to [C][H][W]
      const [, h, w, c] = shape;
      const channels: number[][][] = [];
      for (let ci = 0; ci < c; ci++) {
        const channel: number[][] = [];
        for (let hi = 0; hi < h; hi++) {
          const row: number[] = [];
          for (let wi = 0; wi < w; wi++) {
            row.push(data[hi * w * c + wi * c + ci]);
          }
          channel.push(row);
        }
        channels.push(channel);
      }
      layerActivations[validNames[i]] = channels;
    } else if (shape.length === 2) {
      // Dense: [batch, units] → number[]
      layerActivations[validNames[i]] = Array.from(data);
    }
  }

  // Final output (softmax probabilities)
  const finalTensor = outputs[outputs.length - 1];
  const prediction = Array.from(finalTensor.dataSync() as Float32Array);

  // Dispose all output tensors
  outputs.forEach((t) => t.dispose());
  extractionModel.dispose();

  return { prediction, layerActivations };
}

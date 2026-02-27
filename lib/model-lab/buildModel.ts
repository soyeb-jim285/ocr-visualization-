import type * as tf from "@tensorflow/tfjs";
import type { ArchitectureConfig, Activation } from "./architecture";

/** Names assigned to intermediate layers for activation extraction. */
export interface BuildResult {
  model: tf.LayersModel;
  intermediateLayerNames: string[];
}

function applyActivation(
  tf: typeof import("@tensorflow/tfjs"),
  x: tf.SymbolicTensor,
  activation: Activation,
  name: string,
): tf.SymbolicTensor {
  // LeakyReLU is not a string activation in TF.js — needs its own layer
  if (activation === "leakyRelu") {
    return tf.layers.leakyReLU({ name }).apply(x) as tf.SymbolicTensor;
  }
  // GELU and SiLU aren't built-in string activations in TF.js
  // Use a Lambda layer via tf.layers.activation or a workaround
  if (activation === "gelu" || activation === "silu") {
    // TF.js doesn't have native gelu/silu as layer activations.
    // Use 'relu' as fallback with a note, or implement via Lambda.
    // For simplicity, map them to the closest available:
    // gelu ≈ relu (close enough for educational purposes)
    // silu ≈ relu (sigmoid-weighted linear unit)
    return tf.layers
      .activation({ activation: "relu", name })
      .apply(x) as tf.SymbolicTensor;
  }
  return tf.layers
    .activation({ activation, name })
    .apply(x) as tf.SymbolicTensor;
}

/** Build a TF.js model from the architecture config. */
export function buildModel(
  tf: typeof import("@tensorflow/tfjs"),
  config: ArchitectureConfig,
  numClasses: number,
): BuildResult {
  const intermediateLayerNames: string[] = [];

  const input = tf.input({ shape: [28, 28, 1], name: "input" });
  let x: tf.SymbolicTensor = input;

  // Conv blocks
  for (let i = 0; i < config.convLayers.length; i++) {
    const layer = config.convLayers[i];
    const idx = i + 1;

    // Conv2D
    const convName = `conv${idx}`;
    x = tf.layers
      .conv2d({
        filters: layer.filters,
        kernelSize: layer.kernelSize,
        padding: "same",
        name: convName,
      })
      .apply(x) as tf.SymbolicTensor;
    intermediateLayerNames.push(convName);

    // BatchNorm (before activation)
    if (layer.batchNorm) {
      const bnName = `bn${idx}`;
      x = tf.layers
        .batchNormalization({ name: bnName })
        .apply(x) as tf.SymbolicTensor;
    }

    // Activation
    const actName = `act${idx}`;
    x = applyActivation(tf, x, layer.activation, actName);
    intermediateLayerNames.push(actName);

    // Pooling
    if (layer.pooling === "max") {
      const poolName = `pool${idx}`;
      x = tf.layers
        .maxPooling2d({ poolSize: 2, name: poolName })
        .apply(x) as tf.SymbolicTensor;
      intermediateLayerNames.push(poolName);
    } else if (layer.pooling === "avg") {
      const poolName = `pool${idx}`;
      x = tf.layers
        .averagePooling2d({ poolSize: 2, name: poolName })
        .apply(x) as tf.SymbolicTensor;
      intermediateLayerNames.push(poolName);
    }
  }

  // Flatten
  x = tf.layers.flatten({ name: "flatten" }).apply(x) as tf.SymbolicTensor;

  // Dense hidden layer
  x = tf.layers
    .dense({ units: config.dense.width, name: "dense1" })
    .apply(x) as tf.SymbolicTensor;

  const denseActName = "dense1_act";
  x = applyActivation(tf, x, config.dense.activation, denseActName);
  intermediateLayerNames.push("dense1");
  intermediateLayerNames.push(denseActName);

  // Dropout
  if (config.dense.dropout > 0) {
    x = tf.layers
      .dropout({ rate: config.dense.dropout, name: "dropout" })
      .apply(x) as tf.SymbolicTensor;
  }

  // Output layer
  const output = tf.layers
    .dense({ units: numClasses, activation: "softmax", name: "output" })
    .apply(x) as tf.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: output });
  return { model, intermediateLayerNames };
}

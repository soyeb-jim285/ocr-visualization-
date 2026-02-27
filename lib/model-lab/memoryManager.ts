import type * as tf from "@tensorflow/tfjs";

/**
 * Safely dispose a TF.js LayersModel and all its weight tensors.
 * Returns null for convenient assignment: `model = disposeModel(tf, model)`
 */
export function disposeModel(
  tf: typeof import("@tensorflow/tfjs"),
  model: tf.LayersModel | null,
): null {
  if (!model) return null;
  try {
    model.dispose();
  } catch {
    // Model may already be disposed
  }
  return null;
}

/** Log current TF.js memory usage to console (debug helper). */
export function logMemory(tf: typeof import("@tensorflow/tfjs")): void {
  const mem = tf.memory();
  console.log(
    `[Model Lab] Tensors: ${mem.numTensors}, Bytes: ${(mem.numBytes / 1e6).toFixed(1)}MB`,
  );
}

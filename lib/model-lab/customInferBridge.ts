/**
 * Module-level bridge that allows the main inference pipeline (useInference)
 * to trigger custom model inference without direct coupling.
 *
 * ModelLabSection registers a callback after training completes.
 * useInference calls it after each successful inference.
 */

type CustomInferFn = (tensor: Float32Array) => void;

let _callback: CustomInferFn | null = null;
let _clearCallback: (() => void) | null = null;

/** Called by ModelLabSection when a model is trained. */
export function registerCustomInfer(cb: CustomInferFn, clearCb: () => void) {
  _callback = cb;
  _clearCallback = clearCb;
}

/** Called by ModelLabSection on reset/unmount. */
export function unregisterCustomInfer() {
  _callback = null;
  _clearCallback = null;
}

/** Called by useInference after successful main inference. */
export function triggerCustomInfer(tensor: Float32Array) {
  _callback?.(tensor);
}

/** Called by DrawingCanvas on clear. */
export function triggerCustomClear() {
  _clearCallback?.();
}

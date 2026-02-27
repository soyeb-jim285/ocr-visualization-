import * as ort from "onnxruntime-web";

let cachedSession: ort.InferenceSession | null = null;

/** Initialize ONNX Runtime Web with WASM backend */
async function initOrt() {
  ort.env.wasm.numThreads = 1;
}

/** Load the main ONNX model (multi-output with all intermediate activations) */
export async function loadModel(
  onProgress?: (fraction: number) => void
): Promise<ort.InferenceSession> {
  if (cachedSession) return cachedSession;

  await initOrt();
  onProgress?.(0.1);

  const modelUrl = "/models/combined-cnn/model.onnx";

  onProgress?.(0.5);
  cachedSession = await ort.InferenceSession.create(modelUrl);
  onProgress?.(1);

  return cachedSession;
}

/** Check if the model is loaded */
export function isModelLoaded(): boolean {
  return cachedSession !== null;
}

/** Invalidate the cached session so the next loadModel() re-creates it. */
export function invalidateSession(): void {
  cachedSession = null;
}

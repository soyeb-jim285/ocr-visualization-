import * as ort from "onnxruntime-web";

let cachedSession: ort.InferenceSession | null = null;
let loading: Promise<ort.InferenceSession> | null = null;

/** Initialize ONNX Runtime Web with WASM backend */
async function initOrt() {
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = "error";
}

/** Load the main ONNX model (multi-output with all intermediate activations).
 *  Concurrent callers share one in-flight load. */
export async function loadModel(): Promise<ort.InferenceSession> {
  if (cachedSession) return cachedSession;

  const p = (loading ??= initOrt()
    .then(() => ort.InferenceSession.create("/models/combined-cnn/model.onnx", { logSeverityLevel: 3 }))
    .then((s) => (cachedSession = s))
    .finally(() => {
      if (loading === p) loading = null;
    }));
  return p;
}

/** Check if the model is loaded */
export function isModelLoaded(): boolean {
  return cachedSession !== null;
}

/** Invalidate the cached session so the next loadModel() re-creates it. */
export function invalidateSession(): void {
  cachedSession = null;
  loading = null;
}

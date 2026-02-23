import * as ort from "onnxruntime-web";
import { BYMERGE_MERGED_INDICES } from "./classes";
import {
  INTERMEDIATE_OUTPUTS,
  LAYER_SHAPES,
  nchwToChannels,
  softmax,
} from "./modelUtils";
import type { InferenceResult } from "./predict";

/** Cache for loaded epoch checkpoint sessions */
const sessionCache = new Map<number, ort.InferenceSession>();

/** Currently loading epochs (prevent duplicate loads) */
const loadingEpochs = new Map<number, Promise<ort.InferenceSession>>();

/**
 * Load a checkpoint ONNX model for a specific epoch.
 * These are single-output models (softmax prediction only).
 */
export async function loadEpochModel(
  epoch: number
): Promise<ort.InferenceSession> {
  if (sessionCache.has(epoch)) {
    return sessionCache.get(epoch)!;
  }

  if (loadingEpochs.has(epoch)) {
    return loadingEpochs.get(epoch)!;
  }

  const paddedEpoch = String(epoch).padStart(2, "0");
  const hfBase = process.env.NEXT_PUBLIC_MODEL_BASE_URL
    || "https://huggingface.co/soyeb-jim285/ocr-visualization-models/resolve/main";
  const modelUrl = `${hfBase}/checkpoints/epoch-${paddedEpoch}/model.onnx`;
  const loadPromise = ort.InferenceSession.create(modelUrl)
    .then((session) => {
      sessionCache.set(epoch, session);
      loadingEpochs.delete(epoch);
      return session;
    })
    .catch((err) => {
      loadingEpochs.delete(epoch);
      throw err;
    });

  loadingEpochs.set(epoch, loadPromise);
  return loadPromise;
}

/**
 * Run inference on a specific epoch's model.
 * Returns the softmax prediction (62-element array).
 * Input: Float32Array in NCHW format [1, 1, 28, 28]
 */
export async function predictAtEpoch(
  inputData: Float32Array,
  epoch: number
): Promise<number[]> {
  const session = await loadEpochModel(epoch);
  const inputTensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);
  const results = await session.run({ input: inputTensor });
  const output = results["output"]?.data as Float32Array;
  if (!output) return [];
  // New multi-output models emit raw logits — apply softmax with ByMerge masking
  return softmax(output);
}

/**
 * Run full inference on a specific epoch's model, extracting all intermediate
 * layer activations (same format as runInference in predict.ts).
 * Input: Float32Array in NCHW format [1, 1, 28, 28]
 */
export async function runEpochInference(
  inputData: Float32Array,
  epoch: number
): Promise<InferenceResult> {
  const session = await loadEpochModel(epoch);
  const inputTensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);
  const results = await session.run({ input: inputTensor });

  const layerActivations: Record<string, number[][][] | number[]> = {};

  for (const name of INTERMEDIATE_OUTPUTS) {
    const tensor = results[name];
    if (!tensor) continue;

    const data = tensor.data as Float32Array;
    const shape = LAYER_SHAPES[name];

    if (shape.type === "conv") {
      layerActivations[name] = nchwToChannels(data, shape.c, shape.h, shape.w);
    } else {
      layerActivations[name] = Array.from(data);
    }
  }

  const outputData = results["output"]?.data as Float32Array;
  const prediction = outputData ? softmax(outputData) : [];

  return { prediction, layerActivations };
}

/**
 * Prefetch adjacent epoch models for smooth slider scrubbing.
 */
export function prefetchAdjacentEpochs(currentEpoch: number): void {
  // Prefetch ±3 epochs around current position
  for (let offset = -3; offset <= 3; offset++) {
    const e = currentEpoch + offset;
    if (e >= 0 && e < 50 && !sessionCache.has(e)) {
      loadEpochModel(e).catch(() => {});
    }
  }
}

/** Number of total epochs available */
export const TOTAL_EPOCHS = 50;

let prefetchAllStarted = false;

/**
 * Prefetch ALL epoch models in the background, 3 at a time.
 * Call once when the epoch timeline component mounts.
 * Loads sequentially in small batches to avoid saturating bandwidth.
 */
export function prefetchAllEpochs(
  onProgress?: (loaded: number, total: number) => void
): void {
  if (prefetchAllStarted) return;
  prefetchAllStarted = true;

  const queue = Array.from({ length: TOTAL_EPOCHS }, (_, i) => i)
    .filter((e) => !sessionCache.has(e));

  let loaded = sessionCache.size;
  const total = TOTAL_EPOCHS;

  async function loadBatch() {
    while (queue.length > 0) {
      const batch = queue.splice(0, 3);
      await Promise.allSettled(
        batch.map((e) =>
          loadEpochModel(e)
            .then(() => {
              loaded++;
              onProgress?.(loaded, total);
            })
            .catch(() => {
              loaded++;
              onProgress?.(loaded, total);
            })
        )
      );
    }
  }

  loadBatch();
}

/** Get number of cached sessions */
export function getCachedModelCount(): number {
  return sessionCache.size;
}

// ── Shared inference result cache ──────────────────────────────────────
// Module-level so both EpochPrefetcher and EpochNetworkVisualization share it.

const inferenceResultCache = new Map<number, InferenceResult>();
let cachedInputId: number = 0; // changes when input changes, invalidating cache

export function getInferenceCache() {
  return inferenceResultCache;
}

export function getInferenceCacheInputId() {
  return cachedInputId;
}

/** Clear inference cache (call when input changes) and bump the input id */
export function clearInferenceCache(): number {
  inferenceResultCache.clear();
  return ++cachedInputId;
}

/**
 * Pre-compute epoch inferences for all 50 epochs in the background, 2 at a time.
 * Stores results in the shared module-level cache.
 * Aborts if inputId changes (meaning user drew something new).
 */
export function prefetchAllEpochInferences(
  inputData: Float32Array,
  inputId: number,
  onProgress?: (computed: number, total: number) => void
): void {
  const queue = Array.from({ length: TOTAL_EPOCHS }, (_, i) => i)
    .filter((e) => !inferenceResultCache.has(e));

  let computed = inferenceResultCache.size;

  async function computeBatch() {
    while (queue.length > 0) {
      // Abort if input changed
      if (cachedInputId !== inputId) return;

      const batch = queue.splice(0, 2);
      await Promise.allSettled(
        batch.map((e) =>
          runEpochInference(inputData, e)
            .then((result) => {
              if (cachedInputId !== inputId) return;
              inferenceResultCache.set(e, result);
              computed++;
              onProgress?.(computed, TOTAL_EPOCHS);
            })
            .catch(() => {
              computed++;
              onProgress?.(computed, TOTAL_EPOCHS);
            })
        )
      );
    }
  }

  computeBatch();
}

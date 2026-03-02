"use client";

import { useCallback, useRef } from "react";
import { preprocessCanvas } from "@/lib/model/preprocess";
import { runInference } from "@/lib/model/predict";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";
import { triggerCustomInfer } from "@/lib/model-lab/customInferBridge";

export function useInference() {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancel = useCallback(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
  }, []);

  // Stable reference — reads modelLoaded from the store directly so the
  // callback never goes stale when the hero stage or other props change.
  const infer = useCallback((imageData: ImageData) => {
    if (!useUIStore.getState().modelLoaded) return;

    // Debounce: wait 150ms after last stroke
    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(async () => {
      // Snapshot generation before async work
      const store = useInferenceStore.getState();
      const gen = store.generation;
      store.setIsInferring(true);
      store.setInputImageData(imageData);

      try {
        const { tensor, pixelArray } = preprocessCanvas(imageData);
        if (useInferenceStore.getState().generation !== gen) return;
        store.setInputTensor(pixelArray);

        const t0 = performance.now();
        const { prediction, layerActivations } = await runInference(tensor);
        const t1 = performance.now();
        if (useInferenceStore.getState().generation !== gen) return;
        store.setPrediction(prediction);
        store.setLayerActivations(layerActivations);
        store.setInferenceTimeMs(t1 - t0);

        // Trigger custom model inference AFTER main inference completes.
        // ONNX Runtime WASM backend can't handle concurrent session.run() calls.
        triggerCustomInfer(tensor);
      } catch (error) {
        console.error("Inference failed:", error);
      } finally {
        const current = useInferenceStore.getState();
        if (current.generation === gen) {
          current.setIsInferring(false);
        }
      }
    }, 150);
  }, []);

  return { infer, cancel };
}

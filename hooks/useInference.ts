"use client";

import { useCallback, useRef } from "react";
import { preprocessCanvas } from "@/lib/model/preprocess";
import { runInference } from "@/lib/model/predict";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";

export function useInference() {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const {
    setInputImageData,
    setInputTensor,
    setLayerActivations,
    setPrediction,
    setIsInferring,
  } = useInferenceStore();
  const modelLoaded = useUIStore((s) => s.modelLoaded);

  const infer = useCallback(
    async (imageData: ImageData) => {
      if (!modelLoaded) return;

      // Debounce: wait 150ms after last stroke
      if (debounceRef.current) clearTimeout(debounceRef.current);

      debounceRef.current = setTimeout(async () => {
        setIsInferring(true);
        setInputImageData(imageData);

        try {
          const { tensor, pixelArray } = preprocessCanvas(imageData);
          setInputTensor(pixelArray);

          const { prediction, layerActivations } = await runInference(tensor);
          setPrediction(prediction);
          setLayerActivations(layerActivations);
        } catch (error) {
          console.error("Inference failed:", error);
        } finally {
          setIsInferring(false);
        }
      }, 150);
    },
    [
      modelLoaded,
      setInputImageData,
      setInputTensor,
      setLayerActivations,
      setPrediction,
      setIsInferring,
    ]
  );

  return { infer };
}

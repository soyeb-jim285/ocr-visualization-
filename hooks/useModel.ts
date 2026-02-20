"use client";

import { useEffect, useState } from "react";
import { loadModel } from "@/lib/model/loadModel";
import { useUIStore } from "@/stores/uiStore";

export function useModel() {
  const [error, setError] = useState<string | null>(null);
  const modelLoaded = useUIStore((s) => s.modelLoaded);
  const modelLoadingProgress = useUIStore((s) => s.modelLoadingProgress);

  useEffect(() => {
    if (modelLoaded) return;

    const { setModelLoaded, setModelLoadingProgress } = useUIStore.getState();

    loadModel((progress) => {
      setModelLoadingProgress(progress);
    })
      .then(() => {
        setModelLoaded(true);
        setModelLoadingProgress(1);
      })
      .catch((err) => {
        console.error("Failed to load model:", err);
        setError(err.message);
      });
  }, [modelLoaded]);

  return { modelLoaded, modelLoadingProgress, error };
}

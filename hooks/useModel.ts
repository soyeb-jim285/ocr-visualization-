"use client";

import { useEffect, useState } from "react";
import { loadModel } from "@/lib/model/loadModel";
import { useUIStore } from "@/stores/uiStore";

export function useModel() {
  const [error, setError] = useState<string | null>(null);
  const modelLoaded = useUIStore((s) => s.modelLoaded);

  useEffect(() => {
    if (modelLoaded) return;

    loadModel()
      .then(() => useUIStore.getState().setModelLoaded(true))
      .catch((err) => {
        console.error("Failed to load model:", err);
        setError(err.message);
      });
  }, [modelLoaded]);

  return { modelLoaded, error };
}

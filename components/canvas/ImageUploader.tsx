"use client";

import { useCallback, useRef, useState } from "react";
import { preprocessImage } from "@/lib/model/preprocess";
import { runInference } from "@/lib/model/predict";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";

export function ImageUploader() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const modelLoaded = useUIStore((s) => s.modelLoaded);

  const handleFile = useCallback(
    async (file: File) => {
      if (!modelLoaded) return;
      if (!file.type.startsWith("image/")) return;

      // Access actions via getState() to avoid subscribing to the entire store
      const store = useInferenceStore.getState();
      store.setIsInferring(true);
      try {
        const { tensor, pixelArray } = await preprocessImage(file);
        store.setInputTensor(pixelArray);

        const { prediction, layerActivations } = await runInference(tensor);
        store.setPrediction(prediction);
        store.setLayerActivations(layerActivations);
      } catch (error) {
        console.error("Image processing failed:", error);
      } finally {
        useInferenceStore.getState().setIsInferring(false);
      }
    },
    [modelLoaded]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`flex cursor-pointer items-center gap-2 rounded-lg border border-dashed px-4 py-2 text-sm transition-colors ${
        isDragging
          ? "border-accent-primary bg-accent-primary/10 text-accent-primary"
          : "border-border text-foreground/40 hover:border-foreground/40 hover:text-foreground/60"
      }`}
      onClick={() => inputRef.current?.click()}
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 16 16"
        fill="none"
        className="opacity-60"
      >
        <path
          d="M14 10v3a1 1 0 01-1 1H3a1 1 0 01-1-1v-3M11 5L8 2 5 5M8 2v9"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span>or upload an image</span>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}

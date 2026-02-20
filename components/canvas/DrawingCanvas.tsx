"use client";

import { useCallback } from "react";
import { useDrawingCanvas } from "@/hooks/useDrawingCanvas";
import { useInference } from "@/hooks/useInference";
import { useInferenceStore } from "@/stores/inferenceStore";

const CANVAS_SIZE = 280; // Display size (pixels)
const INTERNAL_SIZE = 280; // Internal resolution

export function DrawingCanvas() {
  const { infer } = useInference();

  const onStrokeEnd = useCallback(
    (imageData: ImageData) => {
      infer(imageData);
    },
    [infer]
  );

  const { canvasRef, clear: rawClear, hasDrawn, startDrawing, draw, stopDrawing } =
    useDrawingCanvas({
      width: INTERNAL_SIZE,
      height: INTERNAL_SIZE,
      lineWidth: 16,
      strokeColor: "#ffffff",
      backgroundColor: "#000000",
      onStrokeEnd,
    });

  const clear = useCallback(() => {
    rawClear();
    useInferenceStore.getState().reset();
  }, [rawClear]);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative overflow-hidden rounded-2xl border-2 border-border bg-black shadow-lg shadow-accent-primary/10">
        <canvas
          ref={canvasRef}
          width={INTERNAL_SIZE}
          height={INTERNAL_SIZE}
          className="cursor-crosshair touch-none"
          style={{
            width: CANVAS_SIZE,
            height: CANVAS_SIZE,
            imageRendering: "auto",
          }}
          onMouseDown={(e) => startDrawing(e.nativeEvent)}
          onMouseMove={(e) => draw(e.nativeEvent)}
          onMouseUp={() => stopDrawing()}
          onMouseLeave={() => stopDrawing()}
          onTouchStart={(e) => {
            e.preventDefault();
            startDrawing(e.nativeEvent);
          }}
          onTouchMove={(e) => {
            e.preventDefault();
            draw(e.nativeEvent);
          }}
          onTouchEnd={() => stopDrawing()}
          aria-label="Drawing canvas for character input"
        />

        {/* Draw hint overlay */}
        {!hasDrawn && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <p className="text-lg text-foreground/30">
              Draw a letter or digit
            </p>
          </div>
        )}
      </div>

      <button
        onClick={clear}
        className="rounded-lg border border-border px-4 py-2 text-sm text-foreground/60 transition-colors hover:border-accent-primary hover:text-foreground"
      >
        Clear
      </button>
    </div>
  );
}

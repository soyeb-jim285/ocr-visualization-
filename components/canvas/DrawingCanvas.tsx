"use client";

import { useCallback, useRef } from "react";
import { useDrawingCanvas } from "@/hooks/useDrawingCanvas";
import { useInference } from "@/hooks/useInference";
import { ImageUploader } from "@/components/canvas/ImageUploader";
import { useInferenceStore } from "@/stores/inferenceStore";

const INTERNAL_SIZE = 280; // Internal resolution

type CanvasVariant = "hero" | "floating";

interface DrawingCanvasProps {
  variant?: CanvasVariant;
  displaySize?: number;
  onFirstDraw?: () => void;
}

export function DrawingCanvas({
  variant = "hero",
  displaySize,
  onFirstDraw,
}: DrawingCanvasProps) {
  const { infer } = useInference();
  const hasFiredFirstDrawRef = useRef(false);

  const canvasSize = displaySize ?? (variant === "hero" ? 320 : 108);
  const lineWidth = variant === "hero" ? 16 : 11;

  const onStrokeEnd = useCallback(
    (imageData: ImageData) => {
      if (!hasFiredFirstDrawRef.current) {
        hasFiredFirstDrawRef.current = true;
        onFirstDraw?.();
      }
      infer(imageData);
    },
    [infer, onFirstDraw]
  );

  const { canvasRef, clear: rawClear, hasDrawn, startDrawing, draw, stopDrawing } =
    useDrawingCanvas({
      width: INTERNAL_SIZE,
      height: INTERNAL_SIZE,
      lineWidth,
      strokeColor: "#ffffff",
      backgroundColor: "#000000",
      onStrokeEnd,
    });

  const clear = useCallback(() => {
    rawClear();
    useInferenceStore.getState().reset();
  }, [rawClear]);

  return (
    <div className={`flex flex-col items-center ${variant === "hero" ? "gap-3" : "gap-1"}`}>
      <div
        className={`relative overflow-hidden border bg-black ${
          variant === "hero"
            ? "rounded-3xl border-border/80 shadow-lg shadow-accent-primary/15"
            : "rounded border-border/50"
        }`}
      >
        <canvas
          ref={canvasRef}
          width={INTERNAL_SIZE}
          height={INTERNAL_SIZE}
          className="cursor-crosshair touch-none"
          style={{
            width: canvasSize,
            height: canvasSize,
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
        {!hasDrawn && variant === "hero" && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <p className="text-lg text-foreground/30">
              Draw a letter or digit
            </p>
          </div>
        )}
      </div>

      {variant === "hero" ? (
        <button
          onClick={clear}
          className="rounded-lg border border-border/80 px-4 py-2 text-sm text-foreground/65 transition-colors hover:border-accent-primary hover:text-foreground"
        >
          Clear
        </button>
      ) : (
        <div className="flex w-full items-center justify-between gap-1">
          <button
            onClick={clear}
            className="inline-flex h-6 items-center rounded-sm border border-border/60 bg-transparent px-2 text-[10px] font-medium text-foreground/60 transition-colors hover:bg-foreground/10 hover:text-foreground"
          >
            Clear
          </button>
          <ImageUploader compact />
        </div>
      )}
    </div>
  );
}

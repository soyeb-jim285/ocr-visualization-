"use client";

import { useCallback, useRef, useState, useEffect } from "react";
import { useDrawingCanvas } from "@/hooks/useDrawingCanvas";
import { useInference } from "@/hooks/useInference";
import { ImageUploader } from "@/components/canvas/ImageUploader";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";
import { triggerCustomClear } from "@/lib/model-lab/customInferBridge";
import { encodePixelsToHash, pixelsToImageData } from "@/lib/shareUrl";

const INTERNAL_SIZE = 280; // Internal resolution

type CanvasVariant = "hero" | "floating";

interface DrawingCanvasProps {
  variant?: CanvasVariant;
  displaySize?: number;
  onFirstDraw?: () => void;
  /** Pre-loaded 28×28 pixel array from shared URL */
  sharedPixels?: number[][] | null;
}

export function DrawingCanvas({
  variant = "hero",
  displaySize,
  onFirstDraw,
  sharedPixels,
}: DrawingCanvasProps) {
  const { infer, cancel: cancelInference } = useInference();
  const hasFiredFirstDrawRef = useRef(false);
  const [shareState, setShareState] = useState<"idle" | "copied">("idle");

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

  const { canvasRef, clear: rawClear, hasDrawn, setHasDrawn, startDrawing, draw, stopDrawing } =
    useDrawingCanvas({
      width: INTERNAL_SIZE,
      height: INTERNAL_SIZE,
      lineWidth,
      strokeColor: "#ffffff",
      backgroundColor: "#000000",
      onStrokeEnd,
    });

  // Load shared pixels onto the canvas once model is ready
  const modelLoaded = useUIStore((s) => s.modelLoaded);
  const sharedLoadedRef = useRef(false);
  useEffect(() => {
    if (!sharedPixels || sharedLoadedRef.current || !modelLoaded) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    sharedLoadedRef.current = true;
    const upscaled = pixelsToImageData(sharedPixels, INTERNAL_SIZE, INTERNAL_SIZE);
    ctx.putImageData(upscaled, 0, 0);
    setHasDrawn(true);

    if (!hasFiredFirstDrawRef.current) {
      hasFiredFirstDrawRef.current = true;
      onFirstDraw?.();
    }
    infer(upscaled);
  }, [sharedPixels, canvasRef, infer, onFirstDraw, setHasDrawn, modelLoaded]);

  const clear = useCallback(() => {
    cancelInference();
    rawClear();
    useInferenceStore.getState().reset();
    triggerCustomClear();
    hasFiredFirstDrawRef.current = false;
    // Clear the hash when clearing the canvas
    if (window.location.hash) history.replaceState(null, "", window.location.pathname);
  }, [rawClear, cancelInference]);

  const share = useCallback(async () => {
    const tensor = useInferenceStore.getState().inputTensor;
    if (!tensor) return;
    const hash = await encodePixelsToHash(tensor);
    history.replaceState(null, "", `#${hash}`);
    await navigator.clipboard.writeText(window.location.href);
    setShareState("copied");
    setTimeout(() => setShareState("idle"), 2000);
  }, []);

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
        <>
          <div className="flex items-center gap-2">
            <button
              onClick={clear}
              className="rounded-lg border border-border/80 px-4 py-2 text-sm text-foreground/65 transition-colors hover:border-accent-primary hover:text-foreground"
            >
              Clear
            </button>
            {hasDrawn && (
              <button
                onClick={share}
                className="rounded-lg border border-border/80 px-4 py-2 text-sm text-foreground/65 transition-colors hover:border-accent-primary hover:text-foreground"
              >
                {shareState === "copied" ? "Copied!" : "Share"}
              </button>
            )}
          </div>
          <p className="text-center font-mono text-[10px] tracking-wider text-foreground/20">
            A–Z &middot; a–z &middot; 0–9 &middot; ক–হ &middot; compound characters
          </p>
        </>
      ) : (
        <div className="flex w-full items-center justify-between gap-1">
          <div className="flex items-center gap-1">
            <button
              onClick={clear}
              className="inline-flex h-6 items-center rounded-sm border border-border/60 bg-transparent px-2 text-[10px] font-medium text-foreground/60 transition-colors hover:bg-foreground/10 hover:text-foreground"
            >
              Clear
            </button>
            {hasDrawn && (
              <button
                onClick={share}
                className="inline-flex h-6 items-center rounded-sm border border-border/60 bg-transparent px-2 text-[10px] font-medium text-foreground/60 transition-colors hover:bg-foreground/10 hover:text-foreground"
                title="Copy share link"
              >
                {shareState === "copied" ? "Copied!" : "Share"}
              </button>
            )}
          </div>
          <ImageUploader compact />
        </div>
      )}
    </div>
  );
}

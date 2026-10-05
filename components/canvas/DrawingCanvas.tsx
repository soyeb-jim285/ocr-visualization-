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
  const [pressed, setPressed] = useState(false);

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

  const isHero = variant === "hero";
  const tickStep = canvasSize / 28;

  const endStroke = useCallback(() => {
    setPressed(false);
    stopDrawing();
  }, [stopDrawing]);

  return (
    <div className={`flex flex-col ${isHero ? "gap-2 sm:gap-3" : "items-center gap-1"}`}>
      {isHero && (
        <div className="flex items-baseline justify-between font-mono text-[10.5px] tracking-[0.06em] text-ink-3">
          <span>FIG. 0 · SPECIMEN</span>
          <span className="hidden min-[420px]:inline">280 × 280 → 28 × 28</span>
        </div>
      )}
      <div
        style={pressed && isHero ? { boxShadow: "0 0 0 1px rgba(143,227,255,0.35), 0 0 28px rgba(143,227,255,0.14)" } : undefined}
        className={`relative self-center overflow-hidden bg-black ${
          isHero ? "rounded-[2px] border border-rule transition-shadow duration-200" : "rounded-[2px] border border-rule-strong"
        }`}
      >
        {/* Glow is CSS-only: canvas pixels feed the model, so ink must stay pure white */}
        <canvas
          ref={canvasRef}
          width={INTERNAL_SIZE}
          height={INTERNAL_SIZE}
          className="block cursor-crosshair touch-none select-none [-webkit-touch-callout:none]"
          style={{
            width: canvasSize,
            height: canvasSize,
            imageRendering: "auto",
          }}
          onPointerDown={(e) => {
            try { e.currentTarget.setPointerCapture(e.pointerId); } catch {}
            setPressed(true);
            startDrawing(e.nativeEvent);
          }}
          onPointerMove={(e) => {
            for (const ev of e.nativeEvent.getCoalescedEvents?.() ?? [e.nativeEvent]) draw(ev);
          }}
          onPointerUp={endStroke}
          onPointerCancel={endStroke}
          onLostPointerCapture={endStroke}
          aria-label="Drawing canvas for character input"
        />

        {isHero && (
          <>
            {/* 28-step ticks along the top and left edge */}
            <div
              aria-hidden
              className="pointer-events-none absolute inset-x-0 top-0 h-1.5"
              style={{ background: `repeating-linear-gradient(90deg, var(--rule) 0 1px, transparent 1px ${tickStep}px)` }}
            />
            <div
              aria-hidden
              className="pointer-events-none absolute inset-y-0 left-0 w-1.5"
              style={{ background: `repeating-linear-gradient(0deg, var(--rule) 0 1px, transparent 1px ${tickStep}px)` }}
            />
            {/* Ink-down underline */}
            <div
              aria-hidden
              className={`pointer-events-none absolute inset-x-0 bottom-0 h-px origin-left bg-phosphor transition-transform duration-200 ${pressed ? "scale-x-100" : "scale-x-0"}`}
            />
            {!hasDrawn && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                <p className="font-serif text-xl italic text-ink-3">Draw a letter or digit</p>
              </div>
            )}
          </>
        )}
      </div>

      {isHero ? (
        <>
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-5">
              <button onClick={clear} className="text-btn py-2">
                Clear
              </button>
              {hasDrawn && (
                <button onClick={share} className="text-btn py-2">
                  {shareState === "copied" ? "Copied!" : "Share"}
                </button>
              )}
            </div>
            <ImageUploader />
          </div>
          <p className="hidden font-mono text-[11px] tracking-[0.04em] text-ink-3 sm:block">
            A–Z &middot; a–z &middot; 0–9 &middot; ক–হ &middot; compound characters
          </p>
        </>
      ) : (
        <div className="flex w-full items-center justify-between gap-1">
          <div className="flex items-center gap-3">
            <button onClick={clear} className="text-btn py-1.5">
              CLEAR
            </button>
            {hasDrawn && (
              <button onClick={share} className="text-btn py-1.5" title="Copy share link">
                {shareState === "copied" ? "COPIED" : "SHARE"}
              </button>
            )}
          </div>
          <ImageUploader compact />
        </div>
      )}
    </div>
  );
}

"use client";

import { motion, useDragControls, type PanInfo } from "framer-motion";
import { useCallback, useEffect, useMemo, useState } from "react";
import { DrawingCanvas } from "@/components/canvas/DrawingCanvas";
import { ImageUploader } from "@/components/canvas/ImageUploader";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export function HeroSection() {
  const dragControls = useDragControls();
  const heroStage = useUIStore((s) => s.heroStage);
  const setHeroStage = useUIStore((s) => s.setHeroStage);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const hasInferenceData = Object.keys(layerActivations).length > 0;

  const [viewport, setViewport] = useState({ w: 0, h: 0 });
  const [customFloatingPos, setCustomFloatingPos] = useState<{ x: number; y: number } | null>(null);

  const isDrawingStage = heroStage === "drawing";
  const isShrinkingStage = heroStage === "shrinking";
  const isRevealedStage = heroStage === "revealed";
  const shouldUseFloatingLayout = !isDrawingStage;

  useEffect(() => {
    const updateViewport = () => {
      setViewport({ w: window.innerWidth, h: window.innerHeight });
    };

    updateViewport();
    window.addEventListener("resize", updateViewport);
    return () => window.removeEventListener("resize", updateViewport);
  }, []);

  useEffect(() => {
    if (hasInferenceData && isDrawingStage) {
      setHeroStage("shrinking");
    }
  }, [hasInferenceData, isDrawingStage, setHeroStage]);

  const expandedCanvasSize = useMemo(() => {
    if (!viewport.w) return 320;
    return clamp(Math.round(viewport.w * 0.45), 250, 360);
  }, [viewport.w]);

  const floatingCanvasSize = useMemo(() => {
    if (!viewport.w) return 108;
    return clamp(Math.round(viewport.w * 0.23), 92, 126);
  }, [viewport.w]);

  const expandedCardWidth = expandedCanvasSize + 44;
  const floatingCardWidth = floatingCanvasSize + 10;
  const floatingCardHeight = floatingCanvasSize + 40;

  const expandedX = viewport.w ? (viewport.w - expandedCardWidth) / 2 : 16;
  const expandedY = viewport.h ? Math.max(72, viewport.h * 0.14) : 92;

  const maxFloatingX = Math.max(12, viewport.w - floatingCardWidth - 12);
  const maxFloatingY = Math.max(12, viewport.h - floatingCardHeight - 12);

  const defaultFloatingPos = useMemo(
    () => ({
      x: clamp(viewport.w - floatingCardWidth - 24, 12, maxFloatingX),
      y: clamp(Math.max(16, viewport.h * 0.13), 12, maxFloatingY),
    }),
    [viewport.w, viewport.h, floatingCardWidth, maxFloatingX, maxFloatingY]
  );

  const floatingPos = customFloatingPos
    ? {
        x: clamp(customFloatingPos.x, 12, maxFloatingX),
        y: clamp(customFloatingPos.y, 12, maxFloatingY),
      }
    : defaultFloatingPos;

  const fullHeroHeight = viewport.h ? Math.max(viewport.h, 700) : 760;

  const handleFirstDraw = useCallback(() => {
    if (isDrawingStage) {
      setHeroStage("shrinking");
    }
  }, [isDrawingStage, setHeroStage]);

  const handleDragEnd = useCallback(
    (_event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
      setCustomFloatingPos({
        x: clamp(floatingPos.x + info.offset.x, 12, maxFloatingX),
        y: clamp(floatingPos.y + info.offset.y, 12, maxFloatingY),
      });
    },
    [floatingPos.x, floatingPos.y, maxFloatingX, maxFloatingY]
  );

  const handleCanvasTransitionComplete = useCallback(() => {
    if (isShrinkingStage) {
      setHeroStage("revealed");
    }
  }, [isShrinkingStage, setHeroStage]);

  return (
    <motion.section
      className="relative overflow-hidden px-4"
      initial={false}
      animate={{
        minHeight: isDrawingStage ? fullHeroHeight : 170,
        paddingTop: isDrawingStage ? 56 : 18,
        paddingBottom: isDrawingStage ? 44 : 10,
      }}
      transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(99,102,241,0.24),transparent_38%),radial-gradient(circle_at_78%_28%,rgba(6,182,212,0.2),transparent_34%),radial-gradient(circle_at_56%_74%,rgba(14,116,144,0.16),transparent_40%)]" />
        <div className="absolute inset-x-0 top-0 h-52 bg-gradient-to-b from-accent-primary/18 to-transparent" />
      </div>

      <div className="relative z-10 mx-auto flex min-h-[62vh] max-w-4xl flex-col items-center justify-end pb-8 text-center">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: isDrawingStage ? 1 : 0, y: isDrawingStage ? 0 : -16 }}
          transition={{ duration: 0.7 }}
          className="mb-2 text-3xl font-semibold tracking-tight sm:text-5xl"
          style={{
            backgroundImage:
              "linear-gradient(125deg, color-mix(in srgb, var(--foreground) 88%, white 12%), var(--accent-tertiary))",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Neural Network X-Ray
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: isDrawingStage ? 1 : 0, y: isDrawingStage ? 0 : -6 }}
          transition={{ delay: 0.15, duration: 0.7 }}
          className="max-w-xl text-sm text-foreground/55 sm:text-base"
        >
          Draw first. As soon as your stroke completes, the canvas shrinks into a floating
          input and the network stage opens in full size.
        </motion.p>

        <motion.p
          initial={false}
          animate={{ opacity: isRevealedStage ? 1 : 0, y: isRevealedStage ? 0 : 16 }}
          transition={{ duration: 0.65 }}
          className="text-xs uppercase tracking-[0.22em] text-foreground/45"
        >
          Drag the mini canvas anywhere while exploring the network
        </motion.p>
      </div>

      <motion.div
        drag={isRevealedStage}
        dragControls={dragControls}
        dragListener={false}
        dragElastic={0.08}
        dragMomentum={false}
        dragConstraints={{ left: 12, top: 12, right: maxFloatingX, bottom: maxFloatingY }}
        onDragEnd={handleDragEnd}
        onAnimationComplete={handleCanvasTransitionComplete}
        initial={false}
        animate={
          shouldUseFloatingLayout
            ? { x: floatingPos.x, y: floatingPos.y, width: floatingCardWidth }
            : { x: expandedX, y: expandedY, width: expandedCardWidth }
        }
        transition={{
          type: "spring",
          stiffness: 145,
          damping: 20,
          mass: 0.86,
        }}
        className="fixed left-0 top-0 z-50"
      >
        <div
          className={`border backdrop-blur-xl ${
            shouldUseFloatingLayout
              ? "rounded-md border-border/50 bg-surface/90 p-1 shadow-lg shadow-black/30"
              : "rounded-[28px] border-accent-primary/35 bg-surface/80 p-3 shadow-[0_26px_90px_rgba(3,8,22,0.65)] sm:p-4"
          }`}
        >
          {shouldUseFloatingLayout ? (
            isRevealedStage && (
              <div
                onPointerDown={(event) => dragControls.start(event)}
                className="mb-1 flex cursor-grab items-center justify-center active:cursor-grabbing"
              >
                <div className="h-0.5 w-6 rounded-full bg-foreground/25" />
              </div>
            )
          ) : (
            <div className="mb-2 flex items-center justify-between px-1">
              <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-foreground/45">
                Step 1
              </span>
              <span className="text-[10px] uppercase tracking-[0.12em] text-foreground/35">
                Model input canvas
              </span>
            </div>
          )}

          <DrawingCanvas
            variant={shouldUseFloatingLayout ? "floating" : "hero"}
            displaySize={shouldUseFloatingLayout ? floatingCanvasSize : expandedCanvasSize}
            onFirstDraw={handleFirstDraw}
          />

          {isDrawingStage && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25, duration: 0.5 }}
              className="mt-3"
            >
              <ImageUploader />
            </motion.div>
          )}
        </div>
      </motion.div>
    </motion.section>
  );
}

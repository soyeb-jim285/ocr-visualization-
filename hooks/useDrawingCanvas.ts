"use client";

import { useRef, useCallback, useEffect, useState } from "react";

interface UseDrawingCanvasOptions {
  width: number;
  height: number;
  lineWidth?: number;
  strokeColor?: string;
  backgroundColor?: string;
  onStrokeEnd?: (imageData: ImageData) => void;
}

export function useDrawingCanvas({
  width,
  height,
  lineWidth = 16,
  strokeColor = "#ffffff",
  backgroundColor = "#000000",
  onStrokeEnd,
}: UseDrawingCanvasOptions) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);
  const lastPoint = useRef<{ x: number; y: number } | null>(null);
  const [hasDrawn, setHasDrawn] = useState(false);

  const getCtx = useCallback(() => {
    return canvasRef.current?.getContext("2d") ?? null;
  }, []);

  const clear = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);
    setHasDrawn(false);
  }, [getCtx, backgroundColor, width, height]);

  const getPoint = useCallback(
    (e: MouseEvent | TouchEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
      const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
      return {
        x: ((clientX - rect.left) / rect.width) * width,
        y: ((clientY - rect.top) / rect.height) * height,
      };
    },
    [width, height]
  );

  const startDrawing = useCallback(
    (e: MouseEvent | TouchEvent) => {
      isDrawing.current = true;
      lastPoint.current = getPoint(e);
      setHasDrawn(true);
    },
    [getPoint]
  );

  const draw = useCallback(
    (e: MouseEvent | TouchEvent) => {
      if (!isDrawing.current) return;
      const ctx = getCtx();
      if (!ctx || !lastPoint.current) return;

      const point = getPoint(e);
      ctx.beginPath();
      ctx.moveTo(lastPoint.current.x, lastPoint.current.y);
      ctx.lineTo(point.x, point.y);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
      lastPoint.current = point;
    },
    [getCtx, getPoint, strokeColor, lineWidth]
  );

  const stopDrawing = useCallback(() => {
    if (!isDrawing.current) return;
    isDrawing.current = false;
    lastPoint.current = null;

    const canvas = canvasRef.current;
    if (canvas && onStrokeEnd) {
      const ctx = canvas.getContext("2d")!;
      onStrokeEnd(ctx.getImageData(0, 0, width, height));
    }
  }, [onStrokeEnd, width, height]);

  // Initialize canvas on mount
  useEffect(() => {
    clear();
  }, [clear]);

  return { canvasRef, clear, hasDrawn, startDrawing, draw, stopDrawing };
}

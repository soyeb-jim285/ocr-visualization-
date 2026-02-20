"use client";

import { useRef, useEffect, useCallback } from "react";

/**
 * Hook for smooth requestAnimationFrame animations.
 * Returns a start/stop/reset interface.
 */
export function useAnimationFrame(
  callback: (deltaTime: number, elapsed: number) => void,
  running: boolean = true
) {
  const rafRef = useRef<number>(0);
  const previousTimeRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);

  const animate = useCallback(
    (time: number) => {
      if (startTimeRef.current === 0) {
        startTimeRef.current = time;
      }
      const deltaTime = time - previousTimeRef.current;
      const elapsed = time - startTimeRef.current;
      previousTimeRef.current = time;

      callback(deltaTime, elapsed);
      rafRef.current = requestAnimationFrame(animate);
    },
    [callback]
  );

  useEffect(() => {
    if (running) {
      previousTimeRef.current = performance.now();
      rafRef.current = requestAnimationFrame(animate);
    }
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [running, animate]);
}

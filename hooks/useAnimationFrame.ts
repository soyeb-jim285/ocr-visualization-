"use client";

import { useRef, useEffect } from "react";

/**
 * Hook for smooth requestAnimationFrame animations.
 * Returns a start/stop/reset interface.
 */
export function useAnimationFrame(
  callback: (deltaTime: number, elapsed: number) => void,
  running: boolean = true
) {
  const rafRef = useRef<number | null>(null);
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!running) {
      return;
    }

    let previousTime = performance.now();
    let startTime = 0;

    const animate = (time: number) => {
      if (startTime === 0) {
        startTime = time;
      }
      const deltaTime = time - previousTime;
      const elapsed = time - startTime;
      previousTime = time;

      callbackRef.current(deltaTime, elapsed);
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [running]);
}

"use client";

import { useRef, useState, useEffect, type ReactNode } from "react";

interface LazySectionProps {
  children: ReactNode;
  /** Section ID for scroll tracking — kept on placeholder so observers can find it */
  id?: string;
  /** Minimum height for the placeholder before the section renders */
  fallbackHeight?: string;
  /** How far before the viewport to start rendering (IntersectionObserver rootMargin) */
  rootMargin?: string;
}

/**
 * Defers rendering of children until the placeholder scrolls near the viewport.
 * Prevents below-fold sections from computing/rendering on initial load.
 * Once rendered, stays rendered permanently.
 */
export function LazySection({
  children,
  id,
  fallbackHeight = "50vh",
  rootMargin = "300px",
}: LazySectionProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [rootMargin]);

  if (!visible) {
    return <div ref={ref} id={id} style={{ minHeight: fallbackHeight }} />;
  }

  return <>{children}</>;
}

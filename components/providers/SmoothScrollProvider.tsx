"use client";

import { MotionConfig } from "framer-motion";

/** Honors prefers-reduced-motion for all Framer Motion animations. */
export function SmoothScrollProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}

"use client";

import { useScrollSection } from "@/hooks/useScrollSection";

/** Invisible component that tracks which section is in view */
export function ScrollTracker() {
  useScrollSection();
  return null;
}

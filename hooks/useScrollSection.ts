"use client";

import { useEffect, useRef } from "react";
import { useUIStore } from "@/stores/uiStore";

export const SECTION_IDS = [
  "hero",
  "pixel-view",
  "convolution",
  "activation",
  "pooling",
  "deeper-layers",
  "fully-connected",
  "softmax",
  "training",
  "neuron-inspector",
] as const;

export function useScrollSection() {
  const setActiveSection = useUIStore((s) => s.setActiveSection);
  const observerRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const index = SECTION_IDS.indexOf(
              entry.target.id as (typeof SECTION_IDS)[number]
            );
            if (index !== -1) setActiveSection(index);
          }
        }
      },
      { threshold: 0.3 }
    );

    SECTION_IDS.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observerRef.current!.observe(el);
    });

    return () => observerRef.current?.disconnect();
  }, [setActiveSection]);
}

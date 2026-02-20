"use client";

import { useEffect, useRef } from "react";
import { useUIStore } from "@/stores/uiStore";

export const SECTION_IDS = [
  "neuron-network",
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
  const rafRef = useRef(0);
  const lastActiveRef = useRef(-1);

  useEffect(() => {
    const update = () => {
      const threshold = window.innerHeight * 0.4;
      let active = 0;

      for (let i = SECTION_IDS.length - 1; i >= 0; i--) {
        const el = document.getElementById(SECTION_IDS[i]);
        if (el && el.getBoundingClientRect().top <= threshold) {
          active = i;
          break;
        }
      }

      if (active !== lastActiveRef.current) {
        lastActiveRef.current = active;
        setActiveSection(active);
      }
    };

    const onScroll = () => {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(update);
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    update(); // initial check

    return () => {
      window.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(rafRef.current);
    };
  }, [setActiveSection]);
}

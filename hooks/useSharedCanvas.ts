"use client";

import { useEffect, useState } from "react";
import { decodeHashToPixels } from "@/lib/shareUrl";

/** On mount, decode shared pixel data from the URL hash fragment (if present). */
export function useSharedCanvas() {
  const [sharedPixels, setSharedPixels] = useState<number[][] | null>(null);

  useEffect(() => {
    const hash = window.location.hash;
    if (!hash || !hash.includes("d=")) return;

    decodeHashToPixels(hash).then((pixels) => {
      if (pixels) setSharedPixels(pixels);
    });
  }, []);

  return sharedPixels;
}

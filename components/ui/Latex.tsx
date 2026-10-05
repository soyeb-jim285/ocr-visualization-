"use client";

import katex from "katex";
import "katex/dist/katex.min.css";

interface LatexProps {
  math: string;
  display?: boolean;
  className?: string;
}

export function Latex({ math, display = false, className }: LatexProps) {
  const html = katex.renderToString(math, {
    displayMode: display,
    throwOnError: false,
  });
  return (
    <span
      className={display ? `block max-w-full overflow-x-auto text-ink-2 ${className ?? ""}` : className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

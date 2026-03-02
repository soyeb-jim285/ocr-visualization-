"use client";

const TECH_STACK = [
  { label: "Next.js 16", href: "https://nextjs.org" },
  { label: "React 19", href: "https://react.dev" },
  { label: "TypeScript", href: "https://typescriptlang.org" },
  { label: "ONNX Runtime", href: "https://onnxruntime.ai" },
  { label: "Tailwind CSS", href: "https://tailwindcss.com" },
  { label: "Zustand", href: "https://zustand.docs.pmnd.rs" },
  { label: "Framer Motion", href: "https://motion.dev" },
  { label: "Three.js", href: "https://threejs.org" },
];

const LINKS = [
  {
    label: "Source Code",
    href: "https://github.com/soyeb-jim285/ocr-visualization",
    icon: (
      <svg viewBox="0 0 16 16" fill="currentColor" className="h-4 w-4">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
      </svg>
    ),
  },
  {
    label: "Model Weights",
    href: "https://huggingface.co/soyeb-jim285/ocr-visualization-models",
    icon: (
      <svg viewBox="0 0 16 16" fill="currentColor" className="h-4 w-4">
        <path d="M8 1a7 7 0 100 14A7 7 0 008 1zM2 8a6 6 0 1112 0A6 6 0 012 8z" />
        <path d="M8 4a1 1 0 011 1v2.586l1.707 1.707a1 1 0 01-1.414 1.414l-2-2A1 1 0 017 8V5a1 1 0 011-1z" />
      </svg>
    ),
  },
];

export function FooterSection() {
  return (
    <footer className="relative mt-8 border-t border-border/40">
      <div
        className="mx-auto px-4 py-12 sm:px-6 sm:py-16"
        style={{ maxWidth: "var(--content-max-width)" }}
      >
        {/* Tech stack */}
        <div className="mb-8">
          <p className="mb-3 text-center font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/35">
            Built with
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            {TECH_STACK.map(({ label, href }) => (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full border border-border/50 px-3 py-1.5 text-xs text-foreground/50 transition-colors hover:border-accent-primary/40 hover:text-foreground/70"
              >
                {label}
              </a>
            ))}
          </div>
        </div>

        {/* Links */}
        <div className="mb-8 flex flex-wrap items-center justify-center gap-4">
          {LINKS.map(({ label, href, icon }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 rounded-lg border border-border/50 px-4 py-2 text-sm text-foreground/60 transition-colors hover:border-accent-primary/50 hover:text-foreground"
            >
              {icon}
              {label}
            </a>
          ))}
        </div>

        {/* Divider + attribution */}
        <div className="flex flex-col items-center gap-2 pt-6">
          <p className="text-xs text-foreground/30">
            Trained on{" "}
            <a
              href="https://www.nist.gov/itl/products-and-services/emnist-dataset"
              target="_blank"
              rel="noopener noreferrer"
              className="underline decoration-foreground/15 underline-offset-2 hover:text-foreground/50"
            >
              EMNIST ByMerge
            </a>
            {" + "}
            <a
              href="https://data.mendeley.com/datasets/hf6sf8zrkc/2"
              target="_blank"
              rel="noopener noreferrer"
              className="underline decoration-foreground/15 underline-offset-2 hover:text-foreground/50"
            >
              BanglaLekha-Isolated
            </a>
            {" "}— 146 character classes, ~980K training images, 75 epochs
          </p>
          <p className="text-xs text-foreground/20">
            Inference runs entirely in your browser via WebAssembly.
          </p>
        </div>
      </div>
    </footer>
  );
}

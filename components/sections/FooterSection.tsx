const LINKS = [
  { label: "Source", href: "https://github.com/soyeb-jim285/ocr-visualization" },
  { label: "Weights", href: "https://huggingface.co/soyeb-jim285/ocr-visualization-models" },
];

export function FooterSection() {
  return (
    <footer className="relative border-t border-rule">
      <div className="mx-auto max-w-[1200px] px-4 md:px-16 min-[1400px]:px-8">
        <div className="grid gap-10 py-24 md:grid-cols-12 md:gap-x-6">
          <div className="md:col-span-5">
            <p className="font-serif text-3xl text-ink">Neural Network X-Ray</p>
            <p className="caption mt-3">
              Inference: in-browser · WASM · no data leaves this tab
            </p>
          </div>

          <div className="md:col-span-4">
            <ul className="caption space-y-1.5">
              <li>
                [1]{" "}
                <a
                  href="https://www.nist.gov/itl/products-and-services/emnist-dataset"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="link-mono"
                >
                  EMNIST ByMerge (Cohen et al., 2017)
                </a>
              </li>
              <li>
                [2]{" "}
                <a
                  href="https://data.mendeley.com/datasets/hf6sf8zrkc/2"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="link-mono"
                >
                  BanglaLekha-Isolated
                </a>
              </li>
            </ul>
            <p className="caption mt-4">146 classes · ~980K images · 75 epochs</p>
          </div>

          <ul className="flex gap-6 md:col-span-3 md:flex-col md:gap-2">
            {LINKS.map(({ label, href }) => (
              <li key={label}>
                <a href={href} target="_blank" rel="noopener noreferrer" className="link-mono">
                  {label} ↗
                </a>
              </li>
            ))}
          </ul>
        </div>

        <p className="caption pb-8">
          Next.js · React · ONNX Runtime Web · Framer Motion · Tailwind
        </p>
        <p className="pb-16 font-serif text-[clamp(1.5rem,3vw,2rem)] italic text-ink-3">
          Every layer, in the open.
        </p>
      </div>

      <div className="border-t border-rule">
        <p className="caption mx-auto max-w-[1200px] px-4 py-4 md:px-16 min-[1400px]:px-8">
          © soyeb-jim285
        </p>
      </div>
    </footer>
  );
}

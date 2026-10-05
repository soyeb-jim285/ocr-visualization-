"use client";

import { useMemo, useRef, useState } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { useInferenceStore } from "@/stores/inferenceStore";
import { Latex } from "@/components/ui/Latex";
import { Reveal } from "@/components/visualizations/FeatureMapGrid";

export function SecondConvSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const [picked, setPicked] = useState<{ maps: unknown; i: number } | null>(null);
  const liveMaps = layerActivations["relu2"] as number[][][] | undefined;
  // default to the highest-energy channel; a manual pick holds only for the current drawing
  const bestFilter = useMemo(() => {
    let best = 0, bestE = -1;
    liveMaps?.forEach((fm, i) => {
      let e = 0;
      for (const r of fm) for (const v of r) e += v > 0 ? v : 0;
      if (e > bestE) { bestE = e; best = i; }
    });
    return best;
  }, [liveMaps]);
  const selectedFilter = picked && picked.maps === liveMaps ? picked.i : bestFilter;
  const previewRef = useRef<HTMLDivElement>(null);
  const setSelectedFilter = (i: number) => {
    setPicked({ maps: liveMaps, i });
    // phones: preview sits a screen above the picker, so bring it into view
    if (window.matchMedia("(max-width: 639px)").matches)
      previewRef.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        block: "center",
      });
  };

  const conv2Maps = layerActivations["conv2"] as number[][][] | undefined;
  const relu2Maps = layerActivations["relu2"] as number[][][] | undefined;

  const beforeRelu = conv2Maps?.[selectedFilter];
  const afterRelu = relu2Maps?.[selectedFilter];

  const numFilters = conv2Maps?.length ?? 64;

  const stats = useMemo(() => {
    if (!beforeRelu) return null;
    const flat = beforeRelu.flat();
    const total = flat.length;
    const afterFlat = afterRelu?.flat() ?? flat.map((v) => Math.max(0, v));
    const activeNeurons = afterFlat.filter((v) => v > 0).length;
    // zeroed derived from the post-ReLU map so zeroed + active === total
    const negCount = total - activeNeurons;
    const negPercent = ((negCount / total) * 100).toFixed(1);
    return { negCount, negPercent, activeNeurons };
  }, [beforeRelu, afterRelu]);

  const deadCount = useMemo(
    () => relu2Maps?.filter((fm) => fm.every((row) => row.every((v) => v === 0))).length ?? 0,
    [relu2Maps],
  );

  return (
    <SectionWrapper id="second-conv" sig="conv2">
      <SectionHeader
        step={4}
        tag="Conv2 · 64 ch · 28×28"
        title="Second Pass: Deeper Patterns"
        subtitle="The first convolution detected simple edges. Now a second convolution layer reads all 32 of those edge maps simultaneously, learning to combine them into more complex features: curves, corners, intersections."
      />

      <div className="grid grid-cols-1 gap-x-6 gap-y-12 lg:grid-cols-12">
        {/* Left: theory text */}
        <Reveal className="min-w-0 space-y-5 lg:col-span-9 lg:col-start-4">
          <p className="prose-body">
            Unlike conv1 which read a single grayscale channel, conv2 reads{" "}
            <em className="text-ink">all 32 feature maps</em> from relu1 at once. Each of its 64
            filters has a 3&times;3&times;32 kernel, computing a weighted
            sum across all input channels at every spatial position. This allows
            it to detect features that require <em className="text-ink">combinations</em> of edges,
            like a curve (horizontal edge meeting a vertical edge).
          </p>

          <div className="formula !text-[0.9em] [scrollbar-width:thin] [scrollbar-color:var(--rule-strong)_transparent]">
            <Latex
              display
              math="O_k(i,j) = \text{ReLU}\!\left(\sum_{c=1}^{32}\sum_{m,n} I_c(i{+}m,\,j{+}n) \cdot K_{k,c}(m,n) + b_k\right)"
            />
            <span className="eq-no">(4)</span>
          </div>

          <ul className="space-y-1 text-sm text-ink-3">
            <li><Latex math="I_c" /> input channel <Latex math="c" /> (from relu1)</li>
            <li><Latex math="K_{k,c}" /> kernel for filter <Latex math="k" />, channel <Latex math="c" /></li>
            <li><Latex math="O_k" /> output feature map <Latex math="k" /></li>
          </ul>

          <p className="prose-body !text-sm">
            The shape transforms:{" "}
            <Latex math="(32, 28, 28) \xrightarrow{\text{conv2 + ReLU}} (64, 28, 28)" />.
            Parameters:{" "}
            <Latex math="64 \times (3 \times 3 \times 32 + 1) = 18{,}496" />
            (BatchNorm, folded into these weights for inference, adds 2 more per channel during training).
            After convolution, ReLU is applied again, zeroing negatives to
            maintain non-linearity. The output is now ready for max pooling,
            which will compress the spatial dimensions in the next step.
          </p>
        </Reveal>

        {/* Right: before / after ReLU */}
        <Reveal delay={0.16} className="min-w-0 lg:col-span-9 lg:col-start-4">
          <div className="figure" ref={previewRef}>
            <div className="flex w-full flex-col items-center justify-center gap-4 sm:flex-row sm:gap-6">
              <HeatFig label="CONV2 OUTPUT" caption="Before ReLU" data={beforeRelu} />
              <Latex
                math="\xrightarrow{\max(0,\,x)}"
                className="hidden text-ink-3 sm:block"
              />
              <Latex math="\downarrow" className="text-ink-3 sm:hidden" />
              <HeatFig label="RELU2 OUTPUT" caption="Ready for pooling" data={afterRelu} />
            </div>

            {stats && (
              <dl className="mt-6 grid grid-cols-3 border-y border-rule py-3 text-center">
                <Stat v={stats.negCount} k="zeroed" />
                <Stat v={`${stats.negPercent}%`} k="sparsity" />
                <Stat v={stats.activeNeurons} k="active" />
              </dl>
            )}
            <p className="figcap">
              <b>FIG. 4.1</b> Filter {String(selectedFilter + 1).padStart(3, "0")} of {numFilters}, same viridis scale, 28×28.
            </p>
          </div>
        </Reveal>
      </div>

      {/* Specimen gallery: all 64 maps */}
      <Reveal className="figure mt-16">
        <div className="mb-5 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
          <p className="font-serif text-xl italic text-ink-2">
            The specimen drawer
          </p>
          <p className="caption">
            Select a filter. Dashed tiles are dead channels
            {relu2Maps ? ` (${deadCount} of ${relu2Maps.length} silent for this drawing)` : ""}.
          </p>
        </div>
        <div className="grid grid-cols-6 justify-items-center gap-x-2 gap-y-3 sm:grid-cols-[repeat(auto-fill,minmax(56px,1fr))]">
          {relu2Maps
            ? relu2Maps.map((fm, i) => (
                <ActivationHeatmap
                  key={i}
                  data={fm}
                  size={56}
                  label={String(i + 1).padStart(3, "0")}
                  onClick={() => setSelectedFilter(i)}
                  selected={i === selectedFilter}
                />
              ))
            : Array.from({ length: numFilters }, (_, i) => (
                <div
                  key={i}
                  aria-hidden
                  className={`tile border border-rule bg-bg-inset ${
                    i === selectedFilter ? "opacity-100" : "opacity-50"
                  }`}
                  data-selected={i === selectedFilter ? "true" : undefined}
                  style={{ width: 56, maxWidth: "100%", aspectRatio: "1" }}
                />
              ))}
        </div>
        <p className="figcap">
          <b>FIG. 4.2</b> All {numFilters} post-ReLU maps of conv2, 28×28 each.
        </p>
      </Reveal>
    </SectionWrapper>
  );
}

function HeatFig({
  label,
  caption,
  data,
}: {
  label: string;
  caption: string;
  data?: number[][];
}) {
  return (
    <div className="flex flex-col items-center gap-2">
      <span className="font-mono text-[11px] tracking-[0.08em] text-sig">{label}</span>
      <div className="flex items-end gap-2">
        <div className="well plate-marks p-1.5">
          {data ? (
            <ActivationHeatmap data={data} size={180} />
          ) : (
            <div className="mx-auto h-[180px] w-[180px]" />
          )}
        </div>
        <div className="ml-2 flex h-[180px] flex-col items-center justify-between font-mono text-[11px] text-ink-3">
          <span>max</span>
          <span className="legend-ramp !h-auto min-h-0 flex-1 my-1" />
          <span>0</span>
        </div>
      </div>
      <span className="caption">{caption}</span>
    </div>
  );
}

function Stat({ v, k }: { v: string | number; k: string }) {
  return (
    <div>
      <dd className="font-mono text-lg tabular-nums text-ink">{v}</dd>
      <dt className="font-mono text-[11px] text-ink-3">{k}</dt>
    </div>
  );
}

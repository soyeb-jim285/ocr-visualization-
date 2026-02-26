"use client";

import { useMemo, useState } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { useInferenceStore } from "@/stores/inferenceStore";
import { Latex } from "@/components/ui/Latex";

export function SecondConvSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const [selectedFilter, setSelectedFilter] = useState(0);

  const conv2Maps = layerActivations["conv2"] as number[][][] | undefined;
  const relu2Maps = layerActivations["relu2"] as number[][][] | undefined;

  const beforeRelu = conv2Maps?.[selectedFilter];
  const afterRelu = relu2Maps?.[selectedFilter];

  const numFilters = conv2Maps?.length ?? 0;
  const hasData = !!beforeRelu;

  const stats = useMemo(() => {
    if (!beforeRelu) return null;
    const flat = beforeRelu.flat();
    const total = flat.length;
    const negCount = flat.filter((v) => v < 0).length;
    const negPercent = ((negCount / total) * 100).toFixed(1);
    const afterFlat = afterRelu?.flat() ?? flat.map((v) => Math.max(0, v));
    const activeNeurons = afterFlat.filter((v) => v > 0).length;
    return { negCount, negPercent, activeNeurons };
  }, [beforeRelu, afterRelu]);

  return (
    <SectionWrapper id="second-conv">
      <SectionHeader
        step={4}
        title="Second Pass: Deeper Patterns"
        subtitle="The first convolution detected simple edges. Now a second convolution layer reads all 64 of those edge maps simultaneously, learning to combine them into more complex features — curves, corners, intersections."
      />

      <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:gap-12">
        {/* Left: theory text */}
        <div className="flex-1 space-y-4 text-center lg:text-left">
          <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
            Unlike conv1 which read a single grayscale channel, conv2 reads{" "}
            <em>all 64 feature maps</em> from relu1 at once. Each of its 128
            filters has a 3&times;3&times;64 kernel — it computes a weighted
            sum across all input channels at every spatial position. This allows
            it to detect features that require <em>combinations</em> of edges,
            like a curve (horizontal edge meeting a vertical edge).
          </p>

          {/* Main equation */}
          <div className="py-3">
            <Latex
              display
              math="O_k(i,j) = \text{ReLU}\!\left(\sum_{c=1}^{64}\sum_{m,n} I_c(i{+}m,\,j{+}n) \cdot K_{k,c}(m,n) + b_k\right)"
            />
          </div>

          {/* Equation legend */}
          <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-foreground/40 lg:justify-start">
            <span><Latex math="I_c" /> — input channel <Latex math="c" /> (from relu1)</span>
            <span><Latex math="K_{k,c}" /> — kernel for filter <Latex math="k" />, channel <Latex math="c" /></span>
            <span><Latex math="O_k" /> — output feature map <Latex math="k" /></span>
          </div>

          <p className="text-sm leading-relaxed text-foreground/45">
            The shape transforms:{" "}
            <Latex math="(64, 28, 28) \xrightarrow{\text{conv2 + ReLU}} (128, 28, 28)" />.
            Parameters:{" "}
            <Latex math="128 \times (3 \times 3 \times 64 + 1) = 73{,}856" />.
            After convolution, ReLU is applied again — zeroing negatives to
            maintain non-linearity. The output is now ready for max pooling,
            which will compress the spatial dimensions in the next step.
          </p>
        </div>

        {/* Right: visualization */}
        <div className="flex w-full shrink-0 flex-col items-center gap-5 lg:w-auto">
          {hasData ? (
            <>
              {/* Before → After ReLU heatmaps */}
              <div className="flex flex-col items-center gap-5 sm:flex-row sm:gap-6">
                <div className="flex flex-col items-center gap-2">
                  <span className="text-xs font-medium text-red-400">
                    Conv2 output
                  </span>
                  <ActivationHeatmap data={beforeRelu} size={120} />
                  <span className="text-[11px] text-foreground/30">
                    Before ReLU
                  </span>
                </div>

                <div className="flex flex-col items-center gap-1">
                  <Latex
                    math="\xrightarrow{\max(0,\,x)}"
                    className="hidden text-foreground/40 sm:block"
                  />
                  <Latex
                    math="\downarrow"
                    className="text-foreground/40 sm:hidden"
                  />
                </div>

                <div className="flex flex-col items-center gap-2">
                  <span className="text-xs font-medium text-green-400">
                    ReLU2 output
                  </span>
                  {afterRelu ? (
                    <ActivationHeatmap data={afterRelu} size={120} />
                  ) : (
                    <div className="flex h-[120px] w-[120px] items-center justify-center rounded-md border border-border bg-surface">
                      <span className="text-xs text-foreground/20">
                        No data
                      </span>
                    </div>
                  )}
                  <span className="text-[11px] text-foreground/30">
                    Ready for pooling
                  </span>
                </div>
              </div>

              {/* Stats — single line */}
              {stats && (
                <div className="flex items-center gap-4 text-sm">
                  <span className="font-mono font-semibold text-red-400">
                    {stats.negCount}
                  </span>
                  <span className="text-foreground/40">zeroed</span>
                  <span className="text-foreground/15">|</span>
                  <span className="font-mono font-semibold text-foreground/60">
                    {stats.negPercent}%
                  </span>
                  <span className="text-foreground/40">sparsity</span>
                  <span className="text-foreground/15">|</span>
                  <span className="font-mono font-semibold text-green-400">
                    {stats.activeNeurons}
                  </span>
                  <span className="text-foreground/40">active</span>
                </div>
              )}
            </>
          ) : (
            <div className="flex h-32 items-center justify-center">
              <p className="text-foreground/30">
                Draw a character above to see activations
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Filter selection: clickable thumbnails — full width */}
      {hasData && (
        <div className="mt-6 space-y-3">
          <p className="text-center text-xs text-foreground/40">
            Select a filter — click any feature map below
          </p>
          <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
            {relu2Maps
              ? relu2Maps.map((fm, i) => (
                  <ActivationHeatmap
                    key={i}
                    data={fm}
                    size={56}
                    label={`#${i + 1}`}
                    onClick={() => setSelectedFilter(i)}
                    selected={i === selectedFilter}
                  />
                ))
              : Array.from({ length: numFilters }, (_, i) => (
                  <div
                    key={i}
                    className={`flex flex-col items-center gap-1 ${
                      i === selectedFilter ? "opacity-100" : "opacity-40"
                    }`}
                  >
                    <div
                      className="border border-border/50"
                      style={{
                        width: 56,
                        height: 56,
                        backgroundColor: "var(--surface)",
                      }}
                    />
                    <span className="text-xs text-foreground/40">
                      #{i + 1}
                    </span>
                  </div>
                ))}
          </div>
        </div>
      )}
    </SectionWrapper>
  );
}

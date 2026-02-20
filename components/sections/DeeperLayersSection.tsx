"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { FeatureMapGrid } from "@/components/visualizations/FeatureMapGrid";
import { useInferenceStore } from "@/stores/inferenceStore";

export function DeeperLayersSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const conv2Maps = layerActivations["conv2"] as number[][][] | undefined;
  const conv3Maps = layerActivations["conv3"] as number[][][] | undefined;
  const pool2Maps = layerActivations["pool2"] as number[][][] | undefined;

  return (
    <SectionWrapper id="deeper-layers">
      <SectionHeader
        step={5}
        title="Going Deeper"
        subtitle="Each layer builds on the previous one. Layer 2 combines basic edges into curves and intersections. Layer 3 detects high-level character parts. Notice how the feature maps become smaller but more abstract — the network is compressing spatial information into meaningful features."
      />

      <div className="flex flex-col gap-10 sm:gap-16">
        {/* Layer 2: 64 filters, 28x28 → pooled to 14x14 */}
        <div>
          <div className="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
            <span className="rounded-full bg-accent-secondary/10 px-3 py-1 text-sm font-medium text-accent-secondary">
              Conv Layer 2
            </span>
            <span className="text-xs text-foreground/40 sm:text-sm">
              64 filters · 28x28 → 14x14 after pooling
            </span>
          </div>
          {conv2Maps ? (
            <FeatureMapGrid
              featureMaps={conv2Maps.slice(0, 32)}
              layerName="conv2"
              columns={8}
              columnsSm={4}
              cellSize={56}
            />
          ) : (
            <div className="flex h-32 items-center justify-center rounded-xl border border-border bg-surface">
              <p className="text-foreground/30">Draw a character to see deeper features</p>
            </div>
          )}
          {conv2Maps && conv2Maps.length > 32 && (
            <p className="mt-2 text-center text-xs text-foreground/30">
              Showing 32 of {conv2Maps.length} feature maps
            </p>
          )}
        </div>

        {/* Layer 3: 128 filters, 14x14 → pooled to 7x7 */}
        <div>
          <div className="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
            <span className="rounded-full bg-accent-tertiary/10 px-3 py-1 text-sm font-medium text-accent-tertiary">
              Conv Layer 3
            </span>
            <span className="text-xs text-foreground/40 sm:text-sm">
              128 filters · 14x14 → 7x7 after pooling
            </span>
          </div>
          {conv3Maps ? (
            <FeatureMapGrid
              featureMaps={conv3Maps.slice(0, 32)}
              layerName="conv3"
              columns={8}
              columnsSm={4}
              cellSize={56}
            />
          ) : (
            <div className="flex h-32 items-center justify-center rounded-xl border border-border bg-surface">
              <p className="text-foreground/30">Draw a character to see deeper features</p>
            </div>
          )}
          {conv3Maps && conv3Maps.length > 32 && (
            <p className="mt-2 text-center text-xs text-foreground/30">
              Showing 32 of {conv3Maps.length} feature maps
            </p>
          )}
        </div>

        {/* Final pooled output: 7x7x128 */}
        {pool2Maps && (
          <div>
            <div className="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
              <span className="rounded-full bg-accent-warning/10 px-3 py-1 text-sm font-medium text-accent-warning">
                After Pooling
              </span>
              <span className="text-xs text-foreground/40 sm:text-sm">
                128 feature maps · 7x7 each
              </span>
            </div>
            <FeatureMapGrid
              featureMaps={pool2Maps.slice(0, 16)}
              layerName="pool2"
              columns={8}
              columnsSm={4}
              cellSize={56}
            />
            <p className="mt-3 text-center text-sm text-foreground/40">
              These compact 7x7 feature maps will be flattened into a single
              vector of {7 * 7 * 128} = 6,272 values for the dense layers.
            </p>
          </div>
        )}
      </div>
    </SectionWrapper>
  );
}

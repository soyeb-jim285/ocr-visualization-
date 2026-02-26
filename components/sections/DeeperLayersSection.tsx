"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { FeatureMapGrid } from "@/components/visualizations/FeatureMapGrid";
import { useInferenceStore } from "@/stores/inferenceStore";
import { Latex } from "@/components/ui/Latex";

export function DeeperLayersSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const conv3Maps = layerActivations["conv3"] as number[][][] | undefined;
  const relu3Maps = layerActivations["relu3"] as number[][][] | undefined;
  const pool2Maps = layerActivations["pool2"] as number[][][] | undefined;

  return (
    <SectionWrapper id="deeper-layers">
      <SectionHeader
        step={6}
        title="Going Deeper: Third Convolution"
        subtitle="After pooling compressed the spatial dimensions to 14×14, a third convolution layer reads all 64 pooled feature maps. It learns to detect high-level character parts — loops, crossbars, serifs — that require combining many simpler patterns."
      />

      {/* Theory introduction */}
      <div className="mb-10 space-y-4 text-center lg:text-left">
        <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
          Each successive layer has a larger <em>receptive field</em> — by layer
          3, each neuron integrates information from a wide region of the
          original input. The third convolution reads all 64 channels from pool1
          and produces 128 new feature maps, followed by ReLU and a second
          pooling step that further compresses spatial dimensions to 7&times;7.
        </p>

        <div className="py-3">
          <Latex
            display
            math="\underbrace{(128,14,14)}_{\text{pool1}} \xrightarrow{\text{conv3}} \underbrace{(256,14,14)}_{\text{conv3}} \xrightarrow{\text{ReLU}} \underbrace{(256,14,14)}_{\text{relu3}} \xrightarrow{\text{pool}} \underbrace{(256,7,7)}_{\text{pool2}}"
          />
        </div>

        <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-foreground/40 lg:justify-start">
          <span>
            Conv3: <Latex math="256 \times (3 \times 3 \times 128 + 1) = 295{,}168" /> params
          </span>
          <span>
            Output: <Latex math="256 \times 7 \times 7 = 12{,}544" /> values
          </span>
        </div>

        <p className="text-sm leading-relaxed text-foreground/45">
          The filter count doubles again — 64, 128, 256 — while pooling halves
          the spatial dimensions. The total information capacity stays roughly
          constant, but shifts from <em>spatial detail</em> to{" "}
          <em>semantic richness</em>. By the final pooling output, the network
          has distilled your 28&times;28 drawing into 256 compact 7&times;7
          feature maps — a dense, abstract representation ready for
          classification.
        </p>
      </div>

      <div className="flex flex-col gap-10 sm:gap-16">
        {/* Conv3: 128 filters, 14x14 */}
        <div>
          <div className="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
            <span className="rounded-full bg-accent-secondary/10 px-3 py-1 text-sm font-medium text-accent-secondary">
              Conv3 Output
            </span>
            <span className="text-xs text-foreground/40 sm:text-sm">
              128 filters &middot; 14&times;14
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
            <div className="viz-empty-state h-32">
              <p className="text-foreground/30">
                Draw a character to see deeper features
              </p>
            </div>
          )}
          {conv3Maps && conv3Maps.length > 32 && (
            <p className="mt-2 text-center text-xs text-foreground/30">
              Showing 32 of {conv3Maps.length} feature maps
            </p>
          )}
        </div>

        {/* ReLU3: 128 filters, 14x14 */}
        {relu3Maps && (
          <div>
            <div className="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
              <span className="rounded-full bg-accent-tertiary/10 px-3 py-1 text-sm font-medium text-accent-tertiary">
                After ReLU
              </span>
              <span className="text-xs text-foreground/40 sm:text-sm">
                128 feature maps &middot; 14&times;14 &middot; negatives zeroed
              </span>
            </div>
            <FeatureMapGrid
              featureMaps={relu3Maps.slice(0, 32)}
              layerName="relu3"
              columns={8}
              columnsSm={4}
              cellSize={56}
            />
            {relu3Maps.length > 32 && (
              <p className="mt-2 text-center text-xs text-foreground/30">
                Showing 32 of {relu3Maps.length} feature maps
              </p>
            )}
          </div>
        )}

        {/* Pool2: 128 filters, 7x7 */}
        {pool2Maps && (
          <div>
            <div className="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
              <span className="rounded-full bg-accent-warning/10 px-3 py-1 text-sm font-medium text-accent-warning">
                After Pooling
              </span>
              <span className="text-xs text-foreground/40 sm:text-sm">
                128 feature maps &middot; 7&times;7 each
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
              These compact 7&times;7 feature maps will be flattened into a
              single vector of{" "}
              <Latex math="7 \times 7 \times 128 = 6{,}272" /> values for the
              dense layers.
            </p>
          </div>
        )}
      </div>
    </SectionWrapper>
  );
}

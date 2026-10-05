"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { FeatureMapGrid, Reveal } from "@/components/visualizations/FeatureMapGrid";
import { useInferenceStore } from "@/stores/inferenceStore";
import { Latex } from "@/components/ui/Latex";

function Placeholder({ n }: { n: number }) {
  return (
    <div className="grid grid-cols-4 justify-items-center gap-x-2 gap-y-3 sm:grid-cols-8">
      {Array.from({ length: n }, (_, i) => (
        <div key={i} className="tile border border-rule bg-bg-inset opacity-50" style={{ width: 56, height: 56 }} />
      ))}
    </div>
  );
}

function Stage({
  fig,
  title,
  meta,
  children,
  note,
}: {
  fig: string;
  title: string;
  meta: string;
  children: React.ReactNode;
  note?: React.ReactNode;
}) {
  return (
    <Reveal className="figure">
      <div className="mb-5 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <p className="font-serif text-xl italic text-ink-2">{title}</p>
        <p className="caption">{meta}</p>
      </div>
      <div className="well plate-marks p-3 sm:p-4">{children}</div>
      <p className="figcap">
        <b>{fig}</b> {note}
      </p>
    </Reveal>
  );
}

export function DeeperLayersSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const conv3Maps = layerActivations["conv3"] as number[][][] | undefined;
  const relu3Maps = layerActivations["relu3"] as number[][][] | undefined;
  const pool2Maps = layerActivations["pool2"] as number[][][] | undefined;

  return (
    <SectionWrapper id="deeper-layers" sig="conv3">
      <SectionHeader
        step={6}
        tag="Conv3 · 128 ch · 14×14"
        title="Going Deeper: Third Convolution"
        subtitle="After pooling compressed the spatial dimensions to 14×14, a third convolution layer reads all 64 pooled feature maps. It learns to detect high-level character parts (loops, crossbars, serifs) that require combining many simpler patterns."
      />

      {/* Theory introduction */}
      <Reveal className="mb-16 grid grid-cols-1 gap-x-6 gap-y-6 lg:grid-cols-12">
        <div className="min-w-0 space-y-5 lg:col-span-9 lg:col-start-4">
          <p className="prose-body">
            Each successive layer has a larger <em className="text-ink">receptive field</em>: by layer
            3, each neuron integrates information from a wide region of the
            original input. The third convolution reads all 64 channels from pool1
            and produces 128 new feature maps, followed by ReLU and a second
            pooling step that further compresses spatial dimensions to 7&times;7.
          </p>
          <ul className="space-y-1 text-sm text-ink-3">
            <li>
              Conv3: <Latex math="128 \times (3 \times 3 \times 64 + 1) = 73{,}856" /> params
            </li>
            <li>
              Output: <Latex math="128 \times 7 \times 7 = 6{,}272" /> values
            </li>
          </ul>
        </div>
        <div className="min-w-0 lg:col-span-9 lg:col-start-4">
          <div className="formula !mt-0 !text-[0.9em] [scrollbar-width:thin] [scrollbar-color:var(--rule-strong)_transparent]">
            <Latex
              display
              math="\underbrace{(64,14,14)}_{\text{pool1}} \xrightarrow{\text{conv3}} \underbrace{(128,14,14)}_{\text{conv3}} \xrightarrow{\text{ReLU}} \underbrace{(128,14,14)}_{\text{relu3}} \xrightarrow{\text{pool}} \underbrace{(128,7,7)}_{\text{pool2}}"
            />
            <span className="eq-no">(6)</span>
          </div>
          <p className="callout">
            <span className="tag">NOTE</span>
            Filters double (32, 64, 128) while pooling halves the spatial size.
            Capacity stays roughly constant but shifts from spatial detail to
            semantic richness: your 28&times;28 drawing becomes 128 compact
            7&times;7 maps.
          </p>
        </div>
      </Reveal>

      <div className="flex flex-col gap-14 sm:gap-16">
        <Stage
          fig="FIG. 6.1"
          title="Conv3 output"
          meta="128 filters · 14×14"
          note={
            conv3Maps && conv3Maps.length > 32
              ? `Showing 32 of ${conv3Maps.length} feature maps. Tap one to enlarge.`
              : "Tap a map to enlarge."
          }
        >
          {conv3Maps ? (
            <FeatureMapGrid featureMaps={conv3Maps.slice(0, 32)} layerName="conv3" columns={8} columnsSm={5} cellSize={56} />
          ) : (
            <Placeholder n={32} />
          )}
        </Stage>

        <Stage
          fig="FIG. 6.2"
          title="After ReLU"
          meta="128 feature maps · 14×14 · negatives zeroed"
          note={
            relu3Maps && relu3Maps.length > 32
              ? `Showing 32 of ${relu3Maps.length} feature maps. Dashed tiles are silent.`
              : "Dashed tiles are silent."
          }
        >
          {relu3Maps ? (
            <FeatureMapGrid featureMaps={relu3Maps.slice(0, 32)} layerName="relu3" columns={8} columnsSm={5} cellSize={56} />
          ) : (
            <Placeholder n={32} />
          )}
        </Stage>

        <Stage
          fig="FIG. 6.3"
          title="After pooling"
          meta="128 feature maps · 7×7 each"
          note={
            <>
              Flattened into one vector of{" "}
              <Latex math="7 \times 7 \times 128 = 6{,}272" /> values for the
              dense layers.
            </>
          }
        >
          {pool2Maps ? (
            <>
              <FeatureMapGrid featureMaps={pool2Maps.slice(0, 16)} layerName="pool2" columns={8} columnsSm={5} cellSize={56} />
              <p className="caption mt-2">Showing 16 of {pool2Maps.length} feature maps.</p>
            </>
          ) : (
            <Placeholder n={16} />
          )}
        </Stage>
      </div>
    </SectionWrapper>
  );
}

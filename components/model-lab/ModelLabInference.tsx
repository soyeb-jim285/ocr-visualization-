"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import { useModelLabStore } from "@/stores/modelLabStore";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES } from "@/lib/model/classes";
import { INK3, RULE, SIG } from "@/lib/theme";

// Precompute label maps to avoid creating new arrays in selectors
const DIGIT_LABELS = "0123456789".split("");
const EMNIST_62 = EMNIST_CLASSES.slice(0, 62);

const LABEL_MAPS: Record<string, string[]> = {
  digits: DIGIT_LABELS,
  emnist: EMNIST_62,
  bangla: EMNIST_CLASSES, // fallback — actual bangla labels come from dataset
  combined: EMNIST_CLASSES,
};

const customConfig = {
  prob: { label: "Confidence", color: SIG.lab },
} satisfies ChartConfig;

const onnxConfig = {
  prob: { label: "Confidence", color: SIG.dense },
} satisfies ChartConfig;

function getTop5(prediction: number[] | null, labelMap: string[]) {
  if (!prediction) return [];
  const indexed = prediction.map((p, i) => ({
    label: i < labelMap.length ? labelMap[i] : `?${i}`,
    prob: +(p * 100).toFixed(1),
  }));
  indexed.sort((a, b) => b.prob - a.prob);
  return indexed.slice(0, 5);
}

function PredictionChart({
  title,
  data,
  config,
  color,
}: {
  title: string;
  data: { label: string; prob: number }[];
  config: ChartConfig;
  color: string;
}) {
  if (data.length === 0) {
    return (
      <div className="min-w-0 flex-1">
        <h4 className="eyebrow mb-2" style={{ color }}>
          {title}
        </h4>
        <div className="viz-empty-state !min-h-[160px]">Draw something to light this up</div>
      </div>
    );
  }

  return (
    <div className="min-w-0 flex-1">
      <h4 className="eyebrow mb-2" style={{ color }}>
        {title}
      </h4>
      <div className="well plate-marks p-2">
      <ChartContainer
        config={config}
        className="h-[160px] w-full min-w-0 overflow-hidden"
      >
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 40, bottom: 4, left: 4 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={RULE}
            horizontal={false}
          />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fill: INK3, fontSize: 11, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={{ stroke: RULE }}
            tickFormatter={(v: number) => `${v}%`}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fill: "#b4c0ca", fontSize: 11, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            width={30}
          />
          <ChartTooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0];
              return (
                <div className="rounded-[3px] border border-rule-strong bg-bg-raised px-2 py-1">
                  <p className="font-mono text-[11px]" style={{ color }}>
                    {d.payload.label}: {d.value}%
                  </p>
                </div>
              );
            }}
          />
          <Bar
            dataKey="prob"
            fill={color}
            radius={[0, 2, 2, 0]}
          />
        </BarChart>
      </ChartContainer>
      </div>
    </div>
  );
}

export function ModelLabInference() {
  const customPrediction = useModelLabStore((s) => s.customPrediction);
  const datasetType = useModelLabStore((s) => s.datasetType);
  const onnxPrediction = useInferenceStore((s) => s.prediction);

  const customLabelMap = LABEL_MAPS[datasetType] ?? EMNIST_CLASSES;

  const customTop5 = useMemo(
    () => getTop5(customPrediction, customLabelMap),
    [customPrediction, customLabelMap],
  );

  const onnxTop5 = useMemo(
    () => getTop5(onnxPrediction, EMNIST_CLASSES),
    [onnxPrediction],
  );

  return (
    <div className="figure space-y-4">
      <h3 className="font-serif text-xl text-ink">Test your model</h3>
      <p className="font-sans text-sm leading-relaxed text-ink-2">
        Draw on the canvas at the top of the page. Your custom model&apos;s
        predictions appear here next to the pre-trained model.
      </p>

      <div className="flex flex-col gap-4 md:flex-row md:gap-6">
        <PredictionChart
          title="Your Model"
          data={customTop5}
          config={customConfig}
          color={SIG.lab}
        />
        <PredictionChart
          title="Pre-trained (ONNX)"
          data={onnxTop5}
          config={onnxConfig}
          color={SIG.dense}
        />
      </div>
    </div>
  );
}

"use client";

import { useEffect, useMemo, useRef } from "react";
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
import { preprocessCanvas } from "@/lib/model/preprocess";
import { EMNIST_CLASSES } from "@/lib/model/classes";

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
  prob: { label: "Confidence", color: "#6366f1" },
} satisfies ChartConfig;

const onnxConfig = {
  prob: { label: "Confidence", color: "#06b6d4" },
} satisfies ChartConfig;

interface ModelLabInferenceProps {
  onInfer: (inputData: Float32Array) => void;
}

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
      <div className="flex-1">
        <h4 className="mb-2 text-xs font-semibold" style={{ color }}>
          {title}
        </h4>
        <p className="text-xs text-foreground/30">
          Draw something above to compare
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1">
      <h4 className="mb-2 text-xs font-semibold" style={{ color }}>
        {title}
      </h4>
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
            stroke="rgba(255,255,255,0.05)"
            horizontal={false}
          />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fill: "rgba(232,232,237,0.35)", fontSize: 9 }}
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
            tickFormatter={(v: number) => `${v}%`}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fill: "rgba(232,232,237,0.6)", fontSize: 11, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            width={30}
          />
          <ChartTooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0];
              return (
                <div className="rounded-lg border border-border/70 bg-surface-elevated/95 px-3 py-2 shadow-lg backdrop-blur-sm">
                  <p className="font-mono text-xs" style={{ color }}>
                    {d.payload.label}: {d.value}%
                  </p>
                </div>
              );
            }}
          />
          <Bar
            dataKey="prob"
            fill={color}
            radius={[0, 4, 4, 0]}
            opacity={0.7}
          />
        </BarChart>
      </ChartContainer>
    </div>
  );
}

export function ModelLabInference({ onInfer }: ModelLabInferenceProps) {
  const customPrediction = useModelLabStore((s) => s.customPrediction);
  const datasetType = useModelLabStore((s) => s.datasetType);
  const onnxPrediction = useInferenceStore((s) => s.prediction);
  const inputImageData = useInferenceStore((s) => s.inputImageData);
  const prevImageDataRef = useRef<ImageData | null>(null);

  // Run custom inference whenever the main canvas input changes
  useEffect(() => {
    if (!inputImageData || inputImageData === prevImageDataRef.current) return;
    prevImageDataRef.current = inputImageData;

    const { tensor } = preprocessCanvas(inputImageData);
    onInfer(tensor);
  }, [inputImageData, onInfer]);

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
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-foreground/70">
        Test Your Model
      </h3>
      <p className="text-xs text-foreground/35">
        Draw on the canvas at the top of the page — your custom model&apos;s
        predictions will appear here alongside the pre-trained model.
      </p>

      <div className="flex flex-col gap-4 sm:flex-row sm:gap-6">
        <PredictionChart
          title="Your Model"
          data={customTop5}
          config={customConfig}
          color="#6366f1"
        />
        <PredictionChart
          title="Pre-trained (ONNX)"
          data={onnxTop5}
          config={onnxConfig}
          color="#06b6d4"
        />
      </div>
    </div>
  );
}

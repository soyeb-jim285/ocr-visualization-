"use client";

import { forwardRef, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import { useModelLabStore } from "@/stores/modelLabStore";
import { INK3, PHOSPHOR, RULE } from "@/lib/theme";

const WARN = "#ffb347"; // --accent-warning

const lossConfig = {
  loss: { label: "Train Loss", color: WARN },
  valLoss: { label: "Val Loss", color: WARN },
} satisfies ChartConfig;

const accConfig = {
  acc: { label: "Train Acc", color: PHOSPHOR },
  valAcc: { label: "Val Acc", color: PHOSPHOR },
} satisfies ChartConfig;

type Row = { epoch: number; loss: number; valLoss: number; acc: number; valAcc: number };

const tick = { fill: INK3, fontSize: 11, fontFamily: "var(--font-mono), monospace" };

function MetricChart({
  fig,
  title,
  config,
  trainKey,
  valKey,
  color,
  data,
  percent,
}: {
  fig: string;
  title: string;
  config: ChartConfig;
  trainKey: "loss" | "acc";
  valKey: "valLoss" | "valAcc";
  color: string;
  data: Row[];
  percent?: boolean;
}) {
  const last = data[data.length - 1];
  const fmt = (v: number) => (percent ? `${v.toFixed(1)}%` : v.toFixed(4));
  return (
    <div className="min-w-0 flex-1">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <p className="figcap !mt-0">
          <b>{fig}</b> {title}
        </p>
        <p className="readout whitespace-nowrap" style={{ color }}>
          {fmt(last[trainKey])}
        </p>
      </div>
      <div className="well plate-marks overflow-x-auto p-2">
        <ChartContainer config={config} className="h-[200px] min-w-[300px] w-full overflow-hidden">
          <LineChart data={data} margin={{ top: 8, right: 8, bottom: 20, left: 0 }}>
            <CartesianGrid stroke={RULE} strokeOpacity={0.5} vertical={false} />
            <XAxis
              dataKey="epoch"
              tick={tick}
              tickLine={false}
              axisLine={{ stroke: RULE }}
              label={{ value: "Epoch", position: "insideBottom", offset: -10, fill: INK3, fontSize: 11 }}
            />
            <YAxis
              domain={percent ? [0, 100] : undefined}
              tick={tick}
              tickLine={false}
              axisLine={false}
              tickFormatter={percent ? (v: number) => `${v}%` : undefined}
              width={40}
            />
            <ChartTooltip
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                return (
                  <div className="rounded-[3px] border border-rule-strong bg-bg-raised px-2 py-1 font-mono text-[11px] text-ink">
                    <p className="mb-1 text-ink-3">Epoch {label}</p>
                    {payload.map((entry) => (
                      <p key={String(entry.dataKey)}>
                        {config[entry.dataKey as string]?.label}:{" "}
                        {typeof entry.value === "number" ? fmt(entry.value) : entry.value}
                      </p>
                    ))}
                  </div>
                );
              }}
            />
            <Line
              type="monotone"
              dataKey={trainKey}
              stroke={color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3, fill: color }}
            />
            <Line
              type="monotone"
              dataKey={valKey}
              stroke={color}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              activeDot={{ r: 3, fill: color }}
            />
          </LineChart>
        </ChartContainer>
      </div>
      <div className="mt-2 flex gap-4 font-mono text-[11px] text-ink-3">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4" style={{ background: color }} />
          Train
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-4 border-t-2 border-dashed" style={{ borderColor: color }} />
          Val
        </span>
      </div>
    </div>
  );
}

export const TrainingChart = forwardRef<HTMLDivElement>(function TrainingChart(_props, ref) {
  const trainingHistory = useModelLabStore((s) => s.trainingHistory);

  const data = useMemo<Row[]>(
    () =>
      trainingHistory.map((m) => ({
        epoch: m.epoch,
        loss: +m.loss.toFixed(4),
        valLoss: +m.valLoss.toFixed(4),
        acc: +(m.acc * 100).toFixed(1),
        valAcc: +(m.valAcc * 100).toFixed(1),
      })),
    [trainingHistory],
  );

  if (data.length === 0) return null;

  return (
    <div className="figure space-y-4">
      <h3 className="font-serif text-xl text-ink">Training progress</h3>

      <div ref={ref} className="flex w-full flex-col gap-6 bg-bg p-1 md:flex-row">
        <MetricChart
          fig="FIG. 10.1"
          title="Loss"
          config={lossConfig}
          trainKey="loss"
          valKey="valLoss"
          color={WARN}
          data={data}
        />
        <MetricChart
          fig="FIG. 10.2"
          title="Accuracy"
          config={accConfig}
          trainKey="acc"
          valKey="valAcc"
          color={PHOSPHOR}
          data={data}
          percent
        />
      </div>
    </div>
  );
});

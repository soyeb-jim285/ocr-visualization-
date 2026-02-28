"use client";

import { forwardRef, useMemo } from "react";
import {
  LineChart,
  Line,
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

const lossConfig = {
  loss: { label: "Train Loss", color: "#ef4444" },
  valLoss: { label: "Val Loss", color: "#f87171" },
} satisfies ChartConfig;

const accConfig = {
  acc: { label: "Train Acc", color: "#22c55e" },
  valAcc: { label: "Val Acc", color: "#4ade80" },
} satisfies ChartConfig;

export const TrainingChart = forwardRef<HTMLDivElement>(function TrainingChart(_props, ref) {
  const trainingHistory = useModelLabStore((s) => s.trainingHistory);

  const data = useMemo(
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
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-foreground/70">
        Training Progress
      </h3>

      <div ref={ref} className="flex w-full flex-col gap-4 rounded-xl border border-border/50 bg-black/20 p-3 sm:flex-row sm:gap-3 sm:p-4">
        {/* Loss chart */}
        <div className="flex min-w-0 flex-1 flex-col items-center gap-1">
          <span className="text-[11px] font-semibold text-[#ef4444]">Loss</span>
          <ChartContainer
            config={lossConfig}
            className="h-[180px] w-full min-w-0 overflow-hidden sm:h-[200px]"
          >
            <LineChart
              data={data}
              margin={{ top: 8, right: 8, bottom: 20, left: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.05)"
              />
              <XAxis
                dataKey="epoch"
                tick={{ fill: "rgba(232,232,237,0.35)", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                label={{
                  value: "Epoch",
                  position: "insideBottom",
                  offset: -10,
                  fill: "rgba(232,232,237,0.3)",
                  fontSize: 10,
                }}
              />
              <YAxis
                tick={{ fill: "rgba(239,68,68,0.5)", fontSize: 9 }}
                tickLine={false}
                axisLine={{ stroke: "rgba(239,68,68,0.12)" }}
                width={36}
              />
              <ChartTooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="rounded-lg border border-border/70 bg-surface-elevated/95 px-3 py-2 shadow-lg backdrop-blur-sm">
                      <p className="mb-1 font-mono text-xs text-foreground/50">
                        Epoch {label}
                      </p>
                      {payload.map((entry) => (
                        <p
                          key={String(entry.dataKey)}
                          className="font-mono text-xs"
                          style={{ color: entry.color }}
                        >
                          {lossConfig[entry.dataKey as keyof typeof lossConfig]
                            ?.label}
                          :{" "}
                          {typeof entry.value === "number"
                            ? entry.value.toFixed(4)
                            : entry.value}
                        </p>
                      ))}
                    </div>
                  );
                }}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="var(--color-loss)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3, fill: "#ef4444" }}
              />
              <Line
                type="monotone"
                dataKey="valLoss"
                stroke="var(--color-valLoss)"
                strokeWidth={1.5}
                strokeDasharray="6 3"
                dot={false}
                activeDot={{ r: 3, fill: "#f87171" }}
                opacity={0.6}
              />
            </LineChart>
          </ChartContainer>
          <div className="flex gap-3 text-[10px] text-foreground/35">
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-3 rounded bg-[#ef4444]" />
              Train
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-3 rounded border-t border-dashed border-[#f87171]" />
              Val
            </span>
          </div>
        </div>

        {/* Accuracy chart */}
        <div className="flex min-w-0 flex-1 flex-col items-center gap-1">
          <span className="text-[11px] font-semibold text-[#22c55e]">
            Accuracy
          </span>
          <ChartContainer
            config={accConfig}
            className="h-[180px] w-full min-w-0 overflow-hidden sm:h-[200px]"
          >
            <LineChart
              data={data}
              margin={{ top: 8, right: 8, bottom: 20, left: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.05)"
              />
              <XAxis
                dataKey="epoch"
                tick={{ fill: "rgba(232,232,237,0.35)", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                label={{
                  value: "Epoch",
                  position: "insideBottom",
                  offset: -10,
                  fill: "rgba(232,232,237,0.3)",
                  fontSize: 10,
                }}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: "rgba(34,197,94,0.5)", fontSize: 9 }}
                tickLine={false}
                axisLine={{ stroke: "rgba(34,197,94,0.12)" }}
                tickFormatter={(v: number) => `${v}%`}
                width={36}
              />
              <ChartTooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="rounded-lg border border-border/70 bg-surface-elevated/95 px-3 py-2 shadow-lg backdrop-blur-sm">
                      <p className="mb-1 font-mono text-xs text-foreground/50">
                        Epoch {label}
                      </p>
                      {payload.map((entry) => (
                        <p
                          key={String(entry.dataKey)}
                          className="font-mono text-xs"
                          style={{ color: entry.color }}
                        >
                          {accConfig[entry.dataKey as keyof typeof accConfig]
                            ?.label}
                          :{" "}
                          {typeof entry.value === "number"
                            ? `${entry.value.toFixed(1)}%`
                            : entry.value}
                        </p>
                      ))}
                    </div>
                  );
                }}
              />
              <Line
                type="monotone"
                dataKey="acc"
                stroke="var(--color-acc)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3, fill: "#22c55e" }}
              />
              <Line
                type="monotone"
                dataKey="valAcc"
                stroke="var(--color-valAcc)"
                strokeWidth={1.5}
                strokeDasharray="6 3"
                dot={false}
                activeDot={{ r: 3, fill: "#4ade80" }}
                opacity={0.6}
              />
            </LineChart>
          </ChartContainer>
          <div className="flex gap-3 text-[10px] text-foreground/35">
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-3 rounded bg-[#22c55e]" />
              Train
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-3 rounded border-t border-dashed border-[#4ade80]" />
              Val
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

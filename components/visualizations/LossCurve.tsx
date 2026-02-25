"use client";

import { useMemo, useState } from "react";
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

interface LossCurveProps {
  history: {
    loss: number[];
    accuracy: number[];
    val_loss: number[];
    val_accuracy: number[];
  } | null;
}

/* ── Two separate charts: Loss and Accuracy ─────────────────────────── */

const lossConfig = {
  trainLoss: { label: "Train Loss", color: "#ef4444" },
  valLoss: { label: "Val Loss", color: "#f87171" },
} satisfies ChartConfig;

const accConfig = {
  trainAcc: { label: "Train Accuracy", color: "#22c55e" },
  valAcc: { label: "Val Accuracy", color: "#4ade80" },
} satisfies ChartConfig;

export function LossCurve({ history }: LossCurveProps) {
  const [showLoss, setShowLoss] = useState(true);
  const [showAcc, setShowAcc] = useState(true);

  const data = useMemo(() => {
    if (!history) return [];
    return history.loss.map((_, i) => ({
      epoch: i,
      trainLoss: history.loss[i],
      valLoss: history.val_loss[i],
      trainAcc: +(history.accuracy[i] * 100).toFixed(2),
      valAcc: +(history.val_accuracy[i] * 100).toFixed(2),
    }));
  }, [history]);

  if (!history) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-border bg-surface">
        <p className="text-foreground/30">Training data not loaded</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Toggle buttons */}
      <div className="flex gap-3">
        <button
          onClick={() => setShowLoss(!showLoss)}
          className={`rounded-lg border px-3 py-1 text-xs font-medium transition ${
            showLoss
              ? "border-accent-negative/50 bg-accent-negative/10 text-accent-negative"
              : "border-border bg-surface text-foreground/30"
          }`}
        >
          Loss
        </button>
        <button
          onClick={() => setShowAcc(!showAcc)}
          className={`rounded-lg border px-3 py-1 text-xs font-medium transition ${
            showAcc
              ? "border-accent-positive/50 bg-accent-positive/10 text-accent-positive"
              : "border-border bg-surface text-foreground/30"
          }`}
        >
          Accuracy
        </button>
      </div>

      <div className="flex w-full max-w-4xl flex-col gap-6 sm:flex-row sm:gap-4">
        {/* Loss chart */}
        {showLoss && (
          <div className="flex flex-1 flex-col items-center gap-1">
            <span className="text-xs font-semibold text-[#ef4444]">Loss</span>
            <ChartContainer
              config={lossConfig}
              className="h-[240px] w-full rounded-xl border border-border bg-surface sm:h-[280px]"
            >
              <LineChart
                data={data}
                margin={{ top: 12, right: 12, bottom: 24, left: 4 }}
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
                    offset: -12,
                    fill: "rgba(232,232,237,0.3)",
                    fontSize: 10,
                  }}
                />
                <YAxis
                  tick={{ fill: "rgba(239,68,68,0.5)", fontSize: 9 }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(239,68,68,0.12)" }}
                  width={40}
                />
                <ChartTooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    return (
                      <div className="rounded-lg border border-border bg-surface px-3 py-2 shadow-lg">
                        <p className="mb-1 font-mono text-xs text-foreground/50">
                          Epoch {label}
                        </p>
                        {payload.map((entry) => (
                          <p
                            key={String(entry.dataKey)}
                            className="font-mono text-xs"
                            style={{ color: entry.color }}
                          >
                            {lossConfig[entry.dataKey as keyof typeof lossConfig]?.label}:{" "}
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
                  dataKey="trainLoss"
                  stroke="var(--color-trainLoss)"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: "#ef4444" }}
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
            {/* Legend */}
            <div className="flex gap-4 text-[11px] text-foreground/40">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 rounded bg-[#ef4444]" />
                Train
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 rounded border-t border-dashed border-[#f87171]" />
                Validation
              </span>
            </div>
          </div>
        )}

        {/* Accuracy chart */}
        {showAcc && (
          <div className="flex flex-1 flex-col items-center gap-1">
            <span className="text-xs font-semibold text-[#22c55e]">
              Accuracy
            </span>
            <ChartContainer
              config={accConfig}
              className="h-[240px] w-full rounded-xl border border-border bg-surface sm:h-[280px]"
            >
              <LineChart
                data={data}
                margin={{ top: 12, right: 12, bottom: 24, left: 4 }}
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
                    offset: -12,
                    fill: "rgba(232,232,237,0.3)",
                    fontSize: 10,
                  }}
                />
                <YAxis
                  domain={[
                    (dataMin: number) => Math.floor(dataMin / 5) * 5,
                    100,
                  ]}
                  tick={{ fill: "rgba(34,197,94,0.5)", fontSize: 9 }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(34,197,94,0.12)" }}
                  tickFormatter={(v: number) => `${v}%`}
                  width={44}
                />
                <ChartTooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    return (
                      <div className="rounded-lg border border-border bg-surface px-3 py-2 shadow-lg">
                        <p className="mb-1 font-mono text-xs text-foreground/50">
                          Epoch {label}
                        </p>
                        {payload.map((entry) => (
                          <p
                            key={String(entry.dataKey)}
                            className="font-mono text-xs"
                            style={{ color: entry.color }}
                          >
                            {accConfig[entry.dataKey as keyof typeof accConfig]?.label}:{" "}
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
                  dataKey="trainAcc"
                  stroke="var(--color-trainAcc)"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: "#22c55e" }}
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
            {/* Legend */}
            <div className="flex gap-4 text-[11px] text-foreground/40">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 rounded bg-[#22c55e]" />
                Train
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 rounded border-t border-dashed border-[#4ade80]" />
                Validation
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

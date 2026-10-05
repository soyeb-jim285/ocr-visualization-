"use client";

import { useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
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

const LOSS_C = "#ffb347"; // --accent-warning
const ACC_C = "#8fe3ff"; // --phosphor
const AXIS = "#8896a3"; // --ink-3
const GRID = "rgba(170,205,225,0.07)";
const AXIS_LINE = "rgba(170,205,225,0.14)";

const lossConfig = {
  trainLoss: { label: "Train Loss", color: LOSS_C },
  valLoss: { label: "Val Loss", color: LOSS_C },
} satisfies ChartConfig;

const accConfig = {
  trainAcc: { label: "Train Accuracy", color: ACC_C },
  valAcc: { label: "Val Accuracy", color: ACC_C },
} satisfies ChartConfig;

const tick = { fill: AXIS, fontSize: 11, fontFamily: "var(--font-geist-mono)" };

function Legend({ color }: { color: string }) {
  return (
    <div className="flex gap-4 font-mono text-[11px] text-ink-3">
      <span className="flex items-center gap-2">
        <span className="inline-block h-0.5 w-5" style={{ background: color }} />
        Train
      </span>
      <span className="flex items-center gap-2">
        <span
          className="inline-block h-0 w-5 border-t-[1.5px] border-dashed"
          style={{ borderColor: color }}
        />
        Validation
      </span>
    </div>
  );
}

function Tip({
  active,
  payload,
  label,
  cfg,
  fmt,
}: {
  active?: boolean;
  payload?: { dataKey?: unknown; value?: unknown }[];
  label?: unknown;
  cfg: Record<string, { label: string }>;
  fmt: (n: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-[3px] border border-rule-strong bg-bg-raised px-2 py-1 font-mono text-[11px] text-ink">
      <p className="mb-1 text-ink-3">EPOCH {String(label)}</p>
      {payload.map((e) => (
        <p key={String(e.dataKey)}>
          {cfg[String(e.dataKey)]?.label}:{" "}
          {typeof e.value === "number" ? fmt(e.value) : String(e.value)}
        </p>
      ))}
    </div>
  );
}

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
      <div className="viz-empty-state h-64 w-full">
        <p>Training data not loaded</p>
      </div>
    );
  }

  const last = data[data.length - 1];
  const margin = { top: 12, right: 12, bottom: 24, left: 4 };
  const xAxis = (
    <XAxis
      dataKey="epoch"
      tick={tick}
      tickLine={false}
      axisLine={{ stroke: AXIS_LINE }}
      label={{
        value: "Epoch",
        position: "insideBottom",
        offset: -12,
        fill: AXIS,
        fontSize: 11,
      }}
    />
  );

  return (
    <div className="flex w-full flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-2" role="group" aria-label="Series">
          <button
            onClick={() => setShowLoss(!showLoss)}
            aria-pressed={showLoss}
            className="chip"
          >
            Loss
          </button>
          <button
            onClick={() => setShowAcc(!showAcc)}
            aria-pressed={showAcc}
            className="chip"
          >
            Accuracy
          </button>
        </div>
        {last && (
          <p className="font-mono text-[11px] text-ink-3">
            FINAL · loss <span className="text-ink">{last.valLoss.toFixed(3)}</span> · acc{" "}
            <span className="text-ink">{last.valAcc.toFixed(1)}%</span> (val)
          </p>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {showLoss && (
          <div className="well flex min-w-0 flex-col gap-2 p-4">
            <div className="flex items-baseline justify-between">
              <span className="eyebrow" style={{ color: LOSS_C }}>Loss</span>
              <Legend color={LOSS_C} />
            </div>
            <div className="overflow-x-auto scrollbar-none">
              <ChartContainer
                config={lossConfig}
                className="h-[230px] w-full min-w-[320px] sm:h-[280px]"
              >
                <LineChart data={data} margin={margin}>
                  <CartesianGrid strokeDasharray="2 4" stroke={GRID} />
                  {xAxis}
                  <YAxis
                    tick={tick}
                    tickLine={false}
                    axisLine={{ stroke: AXIS_LINE }}
                    width={40}
                  />
                  <ChartTooltip
                    cursor={{ stroke: "rgba(170,205,225,0.3)" }}
                    content={(p) => (
                      <Tip
                        {...p}
                        cfg={lossConfig}
                        fmt={(n) => n.toFixed(4)}
                      />
                    )}
                  />
                  <Line
                    type="monotone"
                    dataKey="trainLoss"
                    stroke={LOSS_C}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: LOSS_C, stroke: "#06080b" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="valLoss"
                    stroke={LOSS_C}
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                    dot={false}
                    activeDot={{ r: 3, fill: LOSS_C, stroke: "#06080b" }}
                  />
                </LineChart>
              </ChartContainer>
            </div>
          </div>
        )}

        {showAcc && (
          <div className="well flex min-w-0 flex-col gap-2 p-4">
            <div className="flex items-baseline justify-between">
              <span className="eyebrow" style={{ color: ACC_C }}>Accuracy</span>
              <Legend color={ACC_C} />
            </div>
            <div className="overflow-x-auto scrollbar-none">
              <ChartContainer
                config={accConfig}
                className="h-[230px] w-full min-w-[320px] sm:h-[280px]"
              >
                <LineChart data={data} margin={margin}>
                  <CartesianGrid strokeDasharray="2 4" stroke={GRID} />
                  {xAxis}
                  <YAxis
                    domain={[
                      (dataMin: number) => Math.floor(dataMin / 5) * 5,
                      100,
                    ]}
                    tick={tick}
                    tickLine={false}
                    axisLine={{ stroke: AXIS_LINE }}
                    tickFormatter={(v: number) => `${v}%`}
                    width={44}
                  />
                  <ChartTooltip
                    cursor={{ stroke: "rgba(170,205,225,0.3)" }}
                    content={(p) => (
                      <Tip
                        {...p}
                        cfg={accConfig}
                        fmt={(n) => `${n.toFixed(1)}%`}
                      />
                    )}
                  />
                  <Line
                    type="monotone"
                    dataKey="trainAcc"
                    stroke={ACC_C}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: ACC_C, stroke: "#06080b" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="valAcc"
                    stroke={ACC_C}
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                    dot={false}
                    activeDot={{ r: 3, fill: ACC_C, stroke: "#06080b" }}
                  />
                </LineChart>
              </ChartContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

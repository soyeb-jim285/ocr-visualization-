"use client";

import { useRef, useState, useMemo } from "react";

interface LossCurveProps {
  history: {
    loss: number[];
    accuracy: number[];
    val_loss: number[];
    val_accuracy: number[];
  } | null;
}

const WIDTH = 800;
const HEIGHT = 400;
const PADDING = { top: 40, right: 70, bottom: 50, left: 70 };
const PLOT_W = WIDTH - PADDING.left - PADDING.right;
const PLOT_H = HEIGHT - PADDING.top - PADDING.bottom;

export function LossCurve({ history }: LossCurveProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredEpoch, setHoveredEpoch] = useState<number | null>(null);
  const [showLoss, setShowLoss] = useState(true);
  const [showAcc, setShowAcc] = useState(true);

  const scales = useMemo(() => {
    if (!history) return null;
    const epochs = history.loss.length;
    const maxLoss = Math.ceil(Math.max(...history.loss, ...history.val_loss) * 10) / 10;
    const minAcc = Math.floor(Math.min(...history.accuracy, ...history.val_accuracy) * 100) / 100;
    return {
      x: (epoch: number) => PADDING.left + (epoch / Math.max(epochs - 1, 1)) * PLOT_W,
      yLoss: (val: number) => PADDING.top + (val / maxLoss) * PLOT_H,
      yAcc: (val: number) => PADDING.top + ((1 - val) / (1 - minAcc)) * PLOT_H,
      maxLoss,
      minAcc,
      epochs,
    };
  }, [history]);

  if (!history || !scales) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-border bg-surface">
        <p className="text-foreground/30">Training data not loaded</p>
      </div>
    );
  }

  const makePath = (data: number[], yFn: (v: number) => number) => {
    return data
      .map((v, i) => `${i === 0 ? "M" : "L"} ${scales.x(i)} ${yFn(v)}`)
      .join(" ");
  };

  const hoverData =
    hoveredEpoch !== null
      ? {
          loss: history.loss[hoveredEpoch],
          valLoss: history.val_loss[hoveredEpoch],
          acc: history.accuracy[hoveredEpoch],
          valAcc: history.val_accuracy[hoveredEpoch],
        }
      : null;

  // Y-axis tick values
  const lossTicks = Array.from({ length: 5 }, (_, i) => (scales.maxLoss * i) / 4);
  const accTicks = Array.from(
    { length: 5 },
    (_, i) => scales.minAcc + ((1 - scales.minAcc) * (4 - i)) / 4
  );

  return (
    <div ref={containerRef} className="flex flex-col items-center gap-4">
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

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full max-w-3xl rounded-xl border border-border bg-surface"
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = ((e.clientX - rect.left) / rect.width) * WIDTH - PADDING.left;
          const epoch = Math.round((x / PLOT_W) * (scales.epochs - 1));
          if (epoch >= 0 && epoch < scales.epochs) {
            setHoveredEpoch(epoch);
          } else {
            setHoveredEpoch(null);
          }
        }}
        onMouseLeave={() => setHoveredEpoch(null)}
      >
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={`grid-${t}`}
            x1={PADDING.left}
            y1={PADDING.top + t * PLOT_H}
            x2={PADDING.left + PLOT_W}
            y2={PADDING.top + t * PLOT_H}
            stroke="#e2e8f0"
            strokeWidth={0.5}
            opacity={0.06}
          />
        ))}

        {/* X-axis epoch ticks */}
        {Array.from({ length: 6 }, (_, i) => Math.round((i * (scales.epochs - 1)) / 5)).map(
          (epoch) => (
            <text
              key={`x-${epoch}`}
              x={scales.x(epoch)}
              y={HEIGHT - PADDING.bottom + 20}
              fill="#e2e8f0"
              opacity={0.35}
              fontSize={10}
              textAnchor="middle"
            >
              {epoch}
            </text>
          )
        )}

        {/* Left Y-axis: Loss */}
        {showLoss &&
          lossTicks.map((val, i) => (
            <text
              key={`yl-${i}`}
              x={PADDING.left - 10}
              y={scales.yLoss(val) + 3}
              fill="#ef4444"
              opacity={0.5}
              fontSize={10}
              textAnchor="end"
            >
              {val.toFixed(2)}
            </text>
          ))}

        {/* Right Y-axis: Accuracy */}
        {showAcc &&
          accTicks.map((val, i) => (
            <text
              key={`ya-${i}`}
              x={WIDTH - PADDING.right + 10}
              y={scales.yAcc(val) + 3}
              fill="#22c55e"
              opacity={0.5}
              fontSize={10}
              textAnchor="start"
            >
              {(val * 100).toFixed(0)}%
            </text>
          ))}

        {/* Loss lines */}
        {showLoss && (
          <>
            <path
              d={makePath(history.loss, scales.yLoss)}
              fill="none"
              stroke="#ef4444"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d={makePath(history.val_loss, scales.yLoss)}
              fill="none"
              stroke="#ef4444"
              strokeWidth={1.5}
              strokeDasharray="6 3"
              opacity={0.5}
            />
          </>
        )}

        {/* Accuracy lines */}
        {showAcc && (
          <>
            <path
              d={makePath(history.accuracy, scales.yAcc)}
              fill="none"
              stroke="#22c55e"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d={makePath(history.val_accuracy, scales.yAcc)}
              fill="none"
              stroke="#22c55e"
              strokeWidth={1.5}
              strokeDasharray="6 3"
              opacity={0.5}
            />
          </>
        )}

        {/* Axis labels */}
        {showLoss && (
          <text
            x={PADDING.left - 10}
            y={PADDING.top - 15}
            fill="#ef4444"
            fontSize={11}
            fontWeight={600}
            textAnchor="end"
          >
            Loss
          </text>
        )}
        {showAcc && (
          <text
            x={WIDTH - PADDING.right + 10}
            y={PADDING.top - 15}
            fill="#22c55e"
            fontSize={11}
            fontWeight={600}
          >
            Accuracy
          </text>
        )}
        <text
          x={WIDTH / 2}
          y={HEIGHT - 5}
          fill="#e2e8f0"
          opacity={0.35}
          fontSize={11}
          textAnchor="middle"
        >
          Epoch
        </text>

        {/* Hover line & dots */}
        {hoveredEpoch !== null && (
          <>
            <line
              x1={scales.x(hoveredEpoch)}
              y1={PADDING.top}
              x2={scales.x(hoveredEpoch)}
              y2={PADDING.top + PLOT_H}
              stroke="#e2e8f0"
              strokeWidth={1}
              opacity={0.15}
            />
            {showLoss && (
              <>
                <circle
                  cx={scales.x(hoveredEpoch)}
                  cy={scales.yLoss(history.loss[hoveredEpoch])}
                  r={4}
                  fill="#ef4444"
                />
                <circle
                  cx={scales.x(hoveredEpoch)}
                  cy={scales.yLoss(history.val_loss[hoveredEpoch])}
                  r={3}
                  fill="#ef4444"
                  opacity={0.5}
                />
              </>
            )}
            {showAcc && (
              <>
                <circle
                  cx={scales.x(hoveredEpoch)}
                  cy={scales.yAcc(history.accuracy[hoveredEpoch])}
                  r={4}
                  fill="#22c55e"
                />
                <circle
                  cx={scales.x(hoveredEpoch)}
                  cy={scales.yAcc(history.val_accuracy[hoveredEpoch])}
                  r={3}
                  fill="#22c55e"
                  opacity={0.5}
                />
              </>
            )}
          </>
        )}

        {/* Legend */}
        <g transform={`translate(${PADDING.left + 10}, ${PADDING.top + 10})`}>
          {showLoss && (
            <>
              <line x1={0} y1={0} x2={18} y2={0} stroke="#ef4444" strokeWidth={2} />
              <text x={23} y={4} fill="#e2e8f0" opacity={0.45} fontSize={10}>
                Train Loss
              </text>
              <line
                x1={0}
                y1={16}
                x2={18}
                y2={16}
                stroke="#ef4444"
                strokeWidth={1.5}
                strokeDasharray="6 3"
                opacity={0.5}
              />
              <text x={23} y={20} fill="#e2e8f0" opacity={0.45} fontSize={10}>
                Val Loss
              </text>
            </>
          )}
          {showAcc && (
            <g transform={`translate(${showLoss ? 110 : 0}, 0)`}>
              <line x1={0} y1={0} x2={18} y2={0} stroke="#22c55e" strokeWidth={2} />
              <text x={23} y={4} fill="#e2e8f0" opacity={0.45} fontSize={10}>
                Train Acc
              </text>
              <line
                x1={0}
                y1={16}
                x2={18}
                y2={16}
                stroke="#22c55e"
                strokeWidth={1.5}
                strokeDasharray="6 3"
                opacity={0.5}
              />
              <text x={23} y={20} fill="#e2e8f0" opacity={0.45} fontSize={10}>
                Val Acc
              </text>
            </g>
          )}
        </g>
      </svg>

      {/* Hover info */}
      {hoverData && hoveredEpoch !== null && (
        <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 rounded-lg border border-border bg-surface px-4 py-2 font-mono text-xs">
          <span className="text-foreground/50">Epoch {hoveredEpoch}</span>
          {showLoss && (
            <span className="text-accent-negative">
              Loss: {hoverData.loss.toFixed(4)} | Val: {hoverData.valLoss.toFixed(4)}
            </span>
          )}
          {showAcc && (
            <span className="text-accent-positive">
              Acc: {(hoverData.acc * 100).toFixed(1)}% | Val: {(hoverData.valAcc * 100).toFixed(1)}%
            </span>
          )}
        </div>
      )}
    </div>
  );
}

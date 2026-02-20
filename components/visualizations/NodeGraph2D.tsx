"use client";

import { useMemo, useRef, useEffect } from "react";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES, BYMERGE_MERGED_INDICES } from "@/lib/model/classes";

const CANVAS_W = 1100;
const CANVAS_H = 680;

export function NodeGraph2D() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const prediction = useInferenceStore((s) => s.prediction);

  const denseActivations = layerActivations["relu4"] as number[] | undefined;

  // Precompute layout
  const layout = useMemo(() => {
    // Dense layer: show 64 sampled neurons in 2 columns
    const denseCount = 64;
    const denseColGap = 30;
    const denseX = 80;
    const denseNodes = Array.from({ length: denseCount }, (_, i) => {
      const col = i < 32 ? 0 : 1;
      const row = i < 32 ? i : i - 32;
      return {
        x: denseX + col * denseColGap,
        y: 60 + row * 18,
        dataIdx: Math.floor((i / denseCount) * 256),
      };
    });

    // Output: only show trained classes (skip merged), laid out in 3 groups
    const outputX = 920;
    const groups = [
      { label: "0-9", start: 0, end: 10 },
      { label: "A-Z", start: 10, end: 36 },
      { label: "a-z", start: 36, end: 62 },
    ];

    const outputNodes: { x: number; y: number; classIdx: number; label: string; groupLabel: string }[] = [];
    let yPos = 40;
    for (const group of groups) {
      for (let i = group.start; i < group.end; i++) {
        if (BYMERGE_MERGED_INDICES.has(i)) continue;
        outputNodes.push({
          x: outputX,
          y: yPos,
          classIdx: i,
          label: EMNIST_CLASSES[i],
          groupLabel: group.label,
        });
        yPos += 13;
      }
      yPos += 12; // gap between groups
    }

    return { denseNodes, outputNodes };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_W * dpr;
    canvas.height = CANVAS_H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    const { denseNodes, outputNodes } = layout;
    const hasData = denseActivations && prediction;

    // --- Connections ---
    if (hasData) {
      // Find top activated dense neurons
      const topDense = denseNodes
        .map((n, i) => ({ i, val: Math.abs(denseActivations[n.dataIdx] ?? 0) }))
        .sort((a, b) => b.val - a.val)
        .slice(0, 12);

      // Find top predicted outputs
      const topOutput = outputNodes
        .map((n, i) => ({ i, val: prediction[n.classIdx] ?? 0 }))
        .sort((a, b) => b.val - a.val)
        .slice(0, 8);

      const maxConn = (topDense[0]?.val ?? 1) * (topOutput[0]?.val ?? 1);

      for (const d of topDense) {
        for (const o of topOutput) {
          const dn = denseNodes[d.i];
          const on = outputNodes[o.i];
          const strength = d.val * o.val;
          const alpha = Math.max(0.02, (strength / maxConn) * 0.4);

          // Bezier curve
          ctx.beginPath();
          const midX = (dn.x + 20 + on.x - 20) / 2;
          ctx.moveTo(dn.x + 14, dn.y);
          ctx.bezierCurveTo(midX, dn.y, midX, on.y, on.x - 20, on.y);
          ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
          ctx.lineWidth = 1 + strength / maxConn;
          ctx.stroke();
        }
      }
    }

    // --- Dense neurons ---
    const maxAct = hasData
      ? Math.max(...denseNodes.map((n) => Math.abs(denseActivations[n.dataIdx] ?? 0)), 1)
      : 1;

    for (const node of denseNodes) {
      const act = hasData ? Math.abs(denseActivations[node.dataIdx] ?? 0) : 0;
      const t = act / maxAct;
      const r = 4 + t * 4;

      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);

      if (t > 0.1) {
        // Indigo glow
        ctx.fillStyle = `rgba(99, 102, 241, ${0.3 + t * 0.7})`;
        // Glow effect
        ctx.shadowColor = "rgba(99, 102, 241, 0.5)";
        ctx.shadowBlur = t * 8;
      } else {
        ctx.fillStyle = "rgba(60, 60, 80, 0.4)";
        ctx.shadowBlur = 0;
      }
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // --- Output neurons ---
    const topIdx = hasData ? prediction.indexOf(Math.max(...prediction)) : -1;

    for (const node of outputNodes) {
      const prob = hasData ? (prediction[node.classIdx] ?? 0) : 0;
      const isTop = node.classIdx === topIdx;
      const barWidth = Math.max(prob * 150, 0);

      // Probability bar
      if (prob > 0.005) {
        ctx.fillStyle = isTop
          ? "rgba(99, 102, 241, 0.8)"
          : `rgba(139, 92, 246, ${0.15 + prob * 0.6})`;
        const barH = 8;
        ctx.beginPath();
        ctx.roundRect(node.x - 14, node.y - barH / 2, barWidth, barH, 3);
        ctx.fill();
      }

      // Neuron dot
      const dotR = isTop ? 5.5 : 3.5;
      ctx.beginPath();
      ctx.arc(node.x - 18, node.y, dotR, 0, Math.PI * 2);
      if (isTop) {
        ctx.fillStyle = "#6366f1";
        ctx.shadowColor = "rgba(99, 102, 241, 0.6)";
        ctx.shadowBlur = 10;
      } else if (prob > 0.01) {
        ctx.fillStyle = `rgba(139, 92, 246, ${0.3 + prob})`;
        ctx.shadowBlur = 0;
      } else {
        ctx.fillStyle = "rgba(60, 60, 80, 0.35)";
        ctx.shadowBlur = 0;
      }
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label
      ctx.font = isTop ? "bold 11px monospace" : "10px monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillStyle = isTop
        ? "#6366f1"
        : prob > 0.01
        ? `rgba(226, 232, 240, ${0.4 + prob})`
        : "rgba(226, 232, 240, 0.18)";
      ctx.fillText(node.label, node.x - 24, node.y);

      // Percentage for significant predictions
      if (prob > 0.01) {
        ctx.font = "9px monospace";
        ctx.textAlign = "left";
        ctx.fillStyle = isTop
          ? "rgba(99, 102, 241, 0.9)"
          : "rgba(226, 232, 240, 0.35)";
        ctx.fillText(
          `${(prob * 100).toFixed(1)}%`,
          node.x - 14 + barWidth + 4,
          node.y
        );
      }
    }

    // --- Layer labels ---
    ctx.shadowBlur = 0;
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(226, 232, 240, 0.45)";
    ctx.fillText("Dense (256 neurons)", 72, 25);
    ctx.fillText("Output (47 classes)", 730, 14);

    // Group labels
    ctx.font = "9px sans-serif";
    ctx.fillStyle = "rgba(226, 232, 240, 0.25)";
    let lastGroup = "";
    for (const node of outputNodes) {
      if (node.groupLabel !== lastGroup) {
        ctx.textAlign = "left";
        ctx.fillText(node.groupLabel, node.x + 110, node.y);
        lastGroup = node.groupLabel;
      }
    }

    // Placeholder
    if (!hasData) {
      ctx.font = "16px sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = "rgba(226, 232, 240, 0.15)";
      ctx.fillText("Draw a character to see connections", CANVAS_W / 2, CANVAS_H / 2);
    }
  }, [layout, denseActivations, prediction]);

  return (
    <div className="flex flex-col items-center gap-4">
      <canvas
        ref={canvasRef}
        style={{ width: CANVAS_W, height: CANVAS_H }}
        className="w-full max-w-4xl rounded-xl border border-border bg-surface"
      />
    </div>
  );
}

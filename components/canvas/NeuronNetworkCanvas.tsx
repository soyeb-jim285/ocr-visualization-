"use client";

import { useRef, useMemo, useEffect, useCallback } from "react";
import {
  LAYERS,
  CONN_COUNT,
  SIGNAL_COUNT,
  connFromLayer,
  connFromNeuron,
  connToLayer,
  connToNeuron,
  sigConnIdx,
  sigProgress,
  sigSpeed,
  sigIntensity,
  computeLayout,
  bezierPointInline,
  _bx,
  _by,
  type HoveredNeuron,
} from "@/lib/network/networkConstants";

export interface NeuronNetworkCanvasProps {
  width: number;
  height: number;
  activationMapRef: React.RefObject<Map<string, number[]>>;
  outputLabelsRef: React.RefObject<string[]>;
  hoveredLayerRef: React.RefObject<number | null>;
  hoveredNeuronRef: React.RefObject<HoveredNeuron | null>;
  waveRef: React.RefObject<number>;
  showSignals?: boolean;
  /** Increment to trigger a redraw when showSignals=false (on-demand mode). */
  drawTrigger?: number;
  onHoverLayer: (li: number | null) => void;
  onHoverNeuron: (n: HoveredNeuron | null) => void;
  onClickLayer: (li: number, neuronIdx: number | null) => void;
}

export function NeuronNetworkCanvas({
  width,
  height,
  activationMapRef,
  outputLabelsRef,
  hoveredLayerRef,
  hoveredNeuronRef,
  waveRef,
  showSignals = true,
  drawTrigger,
  onHoverLayer,
  onHoverNeuron,
  onClickLayer,
}: NeuronNetworkCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dpr = typeof window !== "undefined" ? Math.min(window.devicePixelRatio || 1, 2) : 1;

  const layout = useMemo(() => computeLayout(width, height), [width, height]);
  const layoutRef = useRef(layout);
  layoutRef.current = layout;

  // Pre-compute resolved connection positions for current layout
  const connResolved = useMemo(() => {
    const l = layout;
    const fromXArr = new Float32Array(CONN_COUNT);
    const fromYArr = new Float32Array(CONN_COUNT);
    const toXArr = new Float32Array(CONN_COUNT);
    const toYArr = new Float32Array(CONN_COUNT);
    for (let ci = 0; ci < CONN_COUNT; ci++) {
      const fi = l.lookup[connFromLayer[ci]]?.[connFromNeuron[ci]];
      const ti = l.lookup[connToLayer[ci]]?.[connToNeuron[ci]];
      if (fi !== undefined && ti !== undefined) {
        fromXArr[ci] = l.posX[fi];
        fromYArr[ci] = l.posY[fi];
        toXArr[ci] = l.posX[ti];
        toYArr[ci] = l.posY[ti];
      }
    }
    return { fromXArr, fromYArr, toXArr, toYArr };
  }, [layout]);

  // Shared draw function stored in a ref so both RAF and on-demand paths can call it
  const drawRef = useRef<() => void>(() => {});

  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const activationMap = activationMapRef.current;
      const outputLabels = outputLabelsRef.current;
      const hoveredLayer = hoveredLayerRef.current;
      const hoveredNeuron = hoveredNeuronRef.current;
      const waveProgress = waveRef.current;
      const l = layoutRef.current;
      const hasData = activationMap.size > 0;
      const { fromXArr, fromYArr, toXArr, toYArr } = connResolved;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // --- Connections (batched into 8 alpha buckets = 8 draw calls) ---
      {
        const ALPHA_BUCKETS = 8;
        const bucketPaths: Path2D[] = [];
        for (let b = 0; b < ALPHA_BUCKETS; b++) bucketPaths[b] = new Path2D();

        for (let ci = 0; ci < CONN_COUNT; ci++) {
          const fl = connFromLayer[ci], tl = connToLayer[ci];
          const fn = connFromNeuron[ci], tn = connToNeuron[ci];
          const fromActs = activationMap.get(LAYERS[fl].name);
          const toActs = activationMap.get(LAYERS[tl].name);
          const connStrength = Math.sqrt((fromActs?.[fn] ?? 0) * (toActs?.[tn] ?? 0));

          const fromWave = waveProgress - fl;
          const toWave = waveProgress - tl;
          const waveAlpha = Math.min(
            fromWave < 0 ? 0 : fromWave > 1 ? 1 : fromWave,
            toWave < 0 ? 0 : toWave > 1 ? 1 : toWave,
          );

          let baseAlpha: number;
          if (hasData) {
            baseAlpha = 0.03 + connStrength * 0.55;
            if (hoveredLayer !== null && (fl === hoveredLayer || tl === hoveredLayer)) {
              baseAlpha = 0.15 + connStrength * 0.7;
            }
          } else {
            baseAlpha = hoveredLayer !== null && (fl === hoveredLayer || tl === hoveredLayer) ? 0.08 : 0.025;
          }
          const alpha = hasData ? baseAlpha * waveAlpha : baseAlpha;
          if (alpha < 0.005) continue;

          const bucket = Math.min((alpha * ALPHA_BUCKETS) | 0, ALPHA_BUCKETS - 1);
          const fx = fromXArr[ci], fy = fromYArr[ci], tx = toXArr[ci], ty = toYArr[ci];
          const dx = (tx - fx) * 0.4;
          const p = bucketPaths[bucket];
          p.moveTo(fx, fy);
          p.bezierCurveTo(fx + dx, fy, tx - dx, ty, tx, ty);
        }

        ctx.lineWidth = 0.6;
        for (let b = 0; b < ALPHA_BUCKETS; b++) {
          const a = (b + 0.5) / ALPHA_BUCKETS;
          ctx.strokeStyle = `rgba(140,160,200,${a})`;
          ctx.stroke(bucketPaths[b]);
        }
      }

      // --- Signal highlights (bright segment sliding along connections) ---
      if (showSignals && hasData) {
        const sigPaths: Path2D[] = [];
        const sigAlphas: number[] = [];
        for (let li = 0; li < LAYERS.length; li++) { sigPaths[li] = new Path2D(); sigAlphas[li] = 0; }

        for (let i = 0; i < SIGNAL_COUNT; i++) {
          sigProgress[i] += sigSpeed[i];
          if (sigProgress[i] > 1) {
            sigProgress[i] -= 1;
            sigIntensity[i] = 0.4 + Math.random() * 0.6;

            // Weighted reassignment — try 3 random connections, keep the strongest
            let bestCi = Math.floor(Math.random() * CONN_COUNT);
            let bestW = 0;
            for (let a = 0; a < 3; a++) {
              const ci = Math.floor(Math.random() * CONN_COUNT);
              const fActs = activationMap.get(LAYERS[connFromLayer[ci]].name);
              const tActs = activationMap.get(LAYERS[connToLayer[ci]].name);
              const w = Math.sqrt((fActs?.[connFromNeuron[ci]] ?? 0) * (tActs?.[connToNeuron[ci]] ?? 0));
              if (w > bestW) { bestW = w; bestCi = ci; }
            }
            sigConnIdx[i] = bestCi;
            // Stronger connections → faster firing frequency
            sigSpeed[i] = 0.003 + bestW * 0.012;
          }

          const ci = sigConnIdx[i];
          const fl = connFromLayer[ci];
          const layerWave = waveProgress - fl;
          if (layerWave < 0.5) continue;

          const tl = connToLayer[ci];
          const fromActs = activationMap.get(LAYERS[fl].name);
          const toActs = activationMap.get(LAYERS[tl].name);
          const fVal = fromActs?.[connFromNeuron[ci]] ?? 0;
          const tVal = toActs?.[connToNeuron[ci]] ?? 0;
          const w = Math.sqrt(fVal * tVal);
          if (w < 0.05) continue;

          const fx = fromXArr[ci], fy = fromYArr[ci], tx = toXArr[ci], ty = toYArr[ci];
          const ddx = (tx - fx) * 0.4;
          const cx1 = fx + ddx, cy1 = fy, cx2 = tx - ddx, cy2 = ty;
          const a = w * sigIntensity[i] * 0.7;
          if (a > sigAlphas[tl]) sigAlphas[tl] = a;

          const t0 = sigProgress[i];
          const p = sigPaths[tl];
          bezierPointInline(fx, fy, cx1, cy1, cx2, cy2, tx, ty, Math.min(t0, 1));
          p.moveTo(_bx, _by);
          bezierPointInline(fx, fy, cx1, cy1, cx2, cy2, tx, ty, Math.min(t0 + 0.04, 1));
          p.lineTo(_bx, _by);
          bezierPointInline(fx, fy, cx1, cy1, cx2, cy2, tx, ty, Math.min(t0 + 0.08, 1));
          p.lineTo(_bx, _by);
          bezierPointInline(fx, fy, cx1, cy1, cx2, cy2, tx, ty, Math.min(t0 + 0.12, 1));
          p.lineTo(_bx, _by);
        }

        ctx.lineWidth = 1.8;
        ctx.lineCap = "round";
        for (let li = 0; li < LAYERS.length; li++) {
          if (sigAlphas[li] < 0.01) continue;
          const rgb = LAYERS[li].rgb;
          ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${sigAlphas[li] * 0.8})`;
          ctx.stroke(sigPaths[li]);
        }
      }

      // --- Neurons ---
      {
        const r = l.radius;
        const TAU = 6.2832;
        for (let li = 0; li < LAYERS.length; li++) {
          const layer = LAYERS[li];
          const rgb = layer.rgb;
          const acts = activationMap.get(layer.name);
          const layerWave = hasData ? waveProgress - li : 0;
          const wClamp = layerWave < 0 ? 0 : layerWave > 1 ? 1 : layerWave;
          const isHovered = hoveredLayer === li;
          const startI = l.layerStartIdx[li];
          const count = layer.displayNeurons;
          const hovNeuronInLayer = hoveredNeuron?.layerIdx === li;

          for (let ni = 0; ni < count; ni++) {
            const idx = startI + ni;
            const activation = acts?.[ni] ?? 0;
            const effectiveAct = activation * wClamp;
            const x = l.posX[idx], y = l.posY[idx];

            // Glow
            if (effectiveAct > 0.15) {
              ctx.beginPath();
              ctx.arc(x, y, r * 2.2, 0, TAU);
              ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${effectiveAct * 0.2})`;
              ctx.fill();
            }

            // Body
            ctx.beginPath();
            ctx.arc(x, y, r, 0, TAU);
            if (effectiveAct > 0.01) {
              ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.15 + effectiveAct * 0.85})`;
            } else {
              ctx.fillStyle = isHovered ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)";
            }
            ctx.fill();

            // Border
            const isThis = hovNeuronInLayer && hoveredNeuron?.neuronIdx === ni;
            if (isThis) {
              ctx.strokeStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
              ctx.lineWidth = 2.5;
            } else if (isHovered) {
              ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.6)`;
              ctx.lineWidth = 1.5;
            } else {
              const ba = effectiveAct > 0.01 ? 0.3 + effectiveAct * 0.5 : 0.12;
              ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${ba})`;
              ctx.lineWidth = 0.8;
            }
            ctx.stroke();
          }

          // Output labels (only 10 max)
          if (layer.type === "output") {
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            for (let ni = 0; ni < count; ni++) {
              if (!outputLabels[ni]) continue;
              const idx = startI + ni;
              const activation = acts?.[ni] ?? 0;
              const effectiveAct = activation * wClamp;
              const la = effectiveAct > 0.3 ? 0.4 + effectiveAct * 0.6 : 0.25;
              ctx.fillStyle = `rgba(255,255,255,${la})`;
              ctx.font = effectiveAct > 0.5 ? "bold 10px system-ui,sans-serif" : "10px system-ui,sans-serif";
              ctx.fillText(outputLabels[ni], l.posX[idx] + r + 6, l.posY[idx]);
            }
          }
        }
      }

      // --- Layer labels ---
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let li = 0; li < LAYERS.length; li++) {
        const layer = LAYERS[li];
        const x = l.layerX[li];
        const maxY = l.layerBottomY[li];
        const isHov = hoveredLayer === li;

        ctx.fillStyle = isHov ? layer.color : "rgba(255,255,255,0.4)";
        ctx.font = isHov ? "bold 10px system-ui,sans-serif" : "10px system-ui,sans-serif";
        ctx.fillText(layer.displayName, x, maxY + 16);

        const unit = (layer.type === "conv" || layer.type === "relu" || layer.type === "pool") && layer.name !== "relu4" ? "ch" : "";
        ctx.fillStyle = "rgba(255,255,255,0.15)";
        ctx.font = "8px system-ui,sans-serif";
        ctx.fillText(`${layer.totalNeurons}${unit ? " " + unit : " neurons"}`, x, maxY + 28);

        if (layer.totalNeurons > layer.displayNeurons) {
          ctx.fillStyle = "rgba(255,255,255,0.12)";
          for (let d = 0; d < 3; d++) {
            ctx.beginPath();
            ctx.arc(x - 4 + d * 4, maxY + 8, 1, 0, 6.2832);
            ctx.fill();
          }
        }
      }
    };

    drawRef.current = draw;

    if (showSignals) {
      // Continuous RAF loop for animated signals
      let raf = 0;
      const animate = () => {
        draw();
        raf = requestAnimationFrame(animate);
      };
      raf = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(raf);
    } else {
      // Initial draw for static mode
      draw();
    }
  }, [width, height, dpr, connResolved, showSignals, activationMapRef, outputLabelsRef, hoveredLayerRef, hoveredNeuronRef, waveRef]);

  // On-demand redraw when drawTrigger changes (static mode only)
  useEffect(() => {
    if (!showSignals) {
      drawRef.current();
    }
  }, [showSignals, drawTrigger]);

  // Mouse handling
  const findNearest = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { layer: null as number | null, neuron: null as HoveredNeuron | null };
    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width, scaleY = height / rect.height;
    const mx = (clientX - rect.left) * scaleX, my = (clientY - rect.top) * scaleY;
    const l = layoutRef.current;

    let closestNeuron: HoveredNeuron | null = null;
    let closestLayer: number | null = null;
    let minNeuronDist = 18, minLayerDist = 40;

    for (let i = 0; i < l.totalNeurons; i++) {
      const dx = mx - l.posX[i], dy = my - l.posY[i];
      const absDx = dx < 0 ? -dx : dx;
      if (absDx < minLayerDist) { minLayerDist = absDx; closestLayer = l.posLayerIdx[i]; }
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minNeuronDist) {
        minNeuronDist = dist;
        closestNeuron = {
          layerIdx: l.posLayerIdx[i], neuronIdx: l.posNeuronIdx[i],
          screenX: l.posX[i] / scaleX + rect.left, screenY: l.posY[i] / scaleY + rect.top,
        };
      }
    }
    return { layer: closestLayer, neuron: closestNeuron };
  }, [width, height]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { layer, neuron } = findNearest(e.clientX, e.clientY);
    onHoverLayer(layer);
    onHoverNeuron(neuron);
    // Immediate redraw in static mode so hover highlights appear without RAF
    if (!showSignals) drawRef.current();
  }, [findNearest, onHoverLayer, onHoverNeuron, showSignals]);

  const handleMouseLeave = useCallback(() => {
    onHoverLayer(null);
    onHoverNeuron(null);
    if (!showSignals) drawRef.current();
  }, [onHoverLayer, onHoverNeuron, showSignals]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { layer, neuron } = findNearest(e.clientX, e.clientY);
    if (layer !== null) onClickLayer(layer, neuron?.neuronIdx ?? null);
  }, [findNearest, onClickLayer]);

  return (
    <canvas
      ref={canvasRef}
      width={width * dpr}
      height={height * dpr}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      style={{ width, height, cursor: "pointer" }}
    />
  );
}

"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { useInferenceStore } from "@/stores/inferenceStore";
import { LAYER_CONFIG } from "@/lib/model/layerInfo";
import { EMNIST_CLASSES } from "@/lib/model/classes";

export function NeuronInspectorSection() {
  const selectedNeuron = useInferenceStore((s) => s.selectedNeuron);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const prediction = useInferenceStore((s) => s.prediction);
  const setSelectedNeuron = useInferenceStore((s) => s.setSelectedNeuron);

  const layerMeta = selectedNeuron
    ? LAYER_CONFIG.find((l) => l.name === selectedNeuron.layerName)
    : null;

  // Get the activation value for the selected neuron
  const activationValue = useMemo(() => {
    if (!selectedNeuron || !layerActivations[selectedNeuron.layerName])
      return null;

    const data = layerActivations[selectedNeuron.layerName];

    // For dense/1D layers
    if (Array.isArray(data) && typeof data[0] === "number") {
      return (data as number[])[selectedNeuron.neuronIndex];
    }

    return null;
  }, [selectedNeuron, layerActivations]);

  // Quick-select buttons for interesting neurons
  const quickSelects = useMemo(() => {
    const selects: { layerName: string; neuronIndex: number; label: string }[] =
      [];

    // Top 5 activated dense neurons
    const denseAct = layerActivations["relu4"] as number[] | undefined;
    if (denseAct) {
      const sorted = denseAct
        .map((v, i) => ({ v, i }))
        .sort((a, b) => b.v - a.v)
        .slice(0, 5);
      sorted.forEach((n) =>
        selects.push({
          layerName: "relu4",
          neuronIndex: n.i,
          label: `Dense #${n.i} (${n.v.toFixed(2)})`,
        })
      );
    }

    // Top 3 output neurons
    if (prediction) {
      const sorted = prediction
        .map((v, i) => ({ v, i }))
        .sort((a, b) => b.v - a.v)
        .slice(0, 3);
      sorted.forEach((n) =>
        selects.push({
          layerName: "output",
          neuronIndex: n.i,
          label: `Output "${EMNIST_CLASSES[n.i]}" (${(n.v * 100).toFixed(1)}%)`,
        })
      );
    }

    return selects;
  }, [layerActivations, prediction]);

  return (
    <SectionWrapper id="neuron-inspector">
      <SectionHeader
        step={9}
        title="Inspect Any Neuron"
        subtitle="Click on any neuron from the dense or output layers above, or use the quick-select buttons below. Each neuron has a specific role in the network's decision-making process."
      />

      <div className="flex flex-col items-center gap-8">
        {/* Quick select */}
        <div className="flex flex-wrap justify-center gap-2">
          {quickSelects.map((qs, i) => (
            <button
              key={i}
              onClick={() =>
                setSelectedNeuron({
                  layerName: qs.layerName,
                  neuronIndex: qs.neuronIndex,
                })
              }
              className={`rounded-lg border px-3 py-1.5 text-xs transition-colors ${
                selectedNeuron?.layerName === qs.layerName &&
                selectedNeuron?.neuronIndex === qs.neuronIndex
                  ? "border-accent-primary bg-accent-primary/10 text-accent-primary"
                  : "border-border text-foreground/50 hover:border-foreground/30"
              }`}
            >
              {qs.label}
            </button>
          ))}
          {quickSelects.length === 0 && (
            <p className="text-sm text-foreground/30">
              Draw a character, then click neurons in the node graph above
            </p>
          )}
        </div>

        {/* Inspector panel */}
        {selectedNeuron && layerMeta && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-2xl rounded-2xl border border-accent-primary/20 bg-surface p-4 sm:p-8"
          >
            <div className="mb-6 flex items-start justify-between">
              <div>
                <h3 className="text-xl font-bold text-foreground">
                  {layerMeta.displayName} — Neuron #{selectedNeuron.neuronIndex}
                </h3>
                <p className="mt-1 text-sm text-foreground/50">
                  {layerMeta.description}
                </p>
              </div>
              <button
                onClick={() => setSelectedNeuron(null)}
                className="text-foreground/30 hover:text-foreground/60"
              >
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <path
                    d="M6 6l8 8m0-8l-8 8"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              {/* Activation value */}
              <div className="flex flex-col items-center rounded-xl border border-border bg-background p-4">
                <span className="text-xs text-foreground/40">
                  Activation Value
                </span>
                <span className="mt-2 font-mono text-3xl font-bold text-accent-primary">
                  {activationValue !== null
                    ? activationValue.toFixed(4)
                    : "N/A"}
                </span>
                <span className="mt-1 text-xs text-foreground/30">
                  {activationValue !== null && activationValue > 0
                    ? "Active — this neuron is firing"
                    : activationValue === 0
                    ? "Inactive — zeroed by ReLU"
                    : ""}
                </span>
              </div>

              {/* Layer info */}
              <div className="flex flex-col rounded-xl border border-border bg-background p-4">
                <span className="text-xs text-foreground/40">Layer Info</span>
                <div className="mt-2 space-y-1 font-mono text-sm">
                  <div className="flex justify-between">
                    <span className="text-foreground/50">Type</span>
                    <span className="text-foreground">{layerMeta.type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-foreground/50">Output Shape</span>
                    <span className="text-foreground">
                      [{layerMeta.outputShape.join(", ")}]
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-foreground/50">Index</span>
                    <span className="text-foreground">
                      {selectedNeuron.neuronIndex}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Output neuron special: show class info */}
            {selectedNeuron.layerName === "output" && prediction && (
              <div className="mt-6 rounded-xl border border-accent-secondary/20 bg-accent-secondary/5 p-4">
                <p className="text-sm text-foreground/60">
                  This neuron represents the class{" "}
                  <span className="font-bold text-accent-secondary">
                    "{EMNIST_CLASSES[selectedNeuron.neuronIndex]}"
                  </span>{" "}
                  with a probability of{" "}
                  <span className="font-mono font-bold text-accent-secondary">
                    {(
                      prediction[selectedNeuron.neuronIndex] * 100
                    ).toFixed(2)}
                    %
                  </span>
                </p>
              </div>
            )}
          </motion.div>
        )}
      </div>
    </SectionWrapper>
  );
}

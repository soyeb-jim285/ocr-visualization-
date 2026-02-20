"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { NodeGraph2D } from "@/components/visualizations/NodeGraph2D";

export function FullyConnectedSection() {
  return (
    <SectionWrapper id="fully-connected">
      <SectionHeader
        step={6}
        title="Making Decisions: Dense Layers"
        subtitle="The spatial features are flattened into a single vector of 6,272 values, then compressed to 256 neurons. Each neuron is connected to every input — it sees the entire character at once. The network is now making decisions about what character this is."
      />

      <div className="flex flex-col items-center gap-6">
        <NodeGraph2D />
        <p className="max-w-lg text-center text-sm text-foreground/40">
          Brighter nodes = higher activation. Brighter connections = stronger influence.
          Click any node to inspect it in the Neuron Inspector section below.
        </p>
      </div>
    </SectionWrapper>
  );
}

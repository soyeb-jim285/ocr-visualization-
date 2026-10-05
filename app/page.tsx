import dynamic from "next/dynamic";
import { EpochPrefetcher } from "@/components/EpochPrefetcher";
import { NeuronNetworkSection } from "@/components/sections/NeuronNetworkSection";
import { ModelLabWrapper } from "@/components/sections/ModelLabWrapper";

import { FooterSection } from "@/components/sections/FooterSection";
import { ScrollTracker } from "@/components/ui/ScrollTracker";
import { Header } from "@/components/ui/Header";
import { LazySection } from "@/components/ui/LazySection";

// matches LazySection fallbackHeight so layout does not jump while the chunk loads

const PixelViewSection = dynamic(() =>
  import("@/components/sections/PixelViewSection").then((m) => m.PixelViewSection),
  { loading: () => <div style={{ minHeight: "50svh" }} /> },
);
const ConvolutionSection = dynamic(() =>
  import("@/components/sections/ConvolutionSection").then((m) => m.ConvolutionSection),
  { loading: () => <div style={{ minHeight: "50svh" }} /> },
);
const ActivationSection = dynamic(() =>
  import("@/components/sections/ActivationSection").then((m) => m.ActivationSection),
  { loading: () => <div style={{ minHeight: "50svh" }} /> },
);
const SecondConvSection = dynamic(() =>
  import("@/components/sections/SecondConvSection").then((m) => m.SecondConvSection),
  { loading: () => <div style={{ minHeight: "50svh" }} /> },
);
const PoolingSection = dynamic(() =>
  import("@/components/sections/PoolingSection").then((m) => m.PoolingSection),
  { loading: () => <div style={{ minHeight: "50svh" }} /> },
);
const DeeperLayersSection = dynamic(() =>
  import("@/components/sections/DeeperLayersSection").then((m) => m.DeeperLayersSection),
  { loading: () => <div style={{ minHeight: "50svh" }} /> },
);
const FullyConnectedSection = dynamic(() =>
  import("@/components/sections/FullyConnectedSection").then((m) => m.FullyConnectedSection),
  { loading: () => <div style={{ minHeight: "50svh" }} /> },
);
const SoftmaxSection = dynamic(() =>
  import("@/components/sections/SoftmaxSection").then((m) => m.SoftmaxSection),
  { loading: () => <div style={{ minHeight: "50svh" }} /> },
);
const TrainingSection = dynamic(() =>
  import("@/components/sections/TrainingSection").then((m) => m.TrainingSection),
  { loading: () => <div style={{ minHeight: "50svh" }} /> },
);

export default function Home() {
  return (
    <>
      <EpochPrefetcher />

      <Header />
      <ScrollTracker />
      <main className="relative">
        <NeuronNetworkSection />
        <LazySection id="pixel-view">
          <PixelViewSection />
        </LazySection>
        <LazySection id="convolution">
          <ConvolutionSection />
        </LazySection>
        <LazySection id="activation">
          <ActivationSection />
        </LazySection>
        <LazySection id="second-conv">
          <SecondConvSection />
        </LazySection>
        <LazySection id="pooling">
          <PoolingSection />
        </LazySection>
        <LazySection id="deeper-layers">
          <DeeperLayersSection />
        </LazySection>
        <LazySection id="fully-connected">
          <FullyConnectedSection />
        </LazySection>
        <LazySection id="softmax">
          <SoftmaxSection />
        </LazySection>
        <LazySection id="training">
          <TrainingSection />
        </LazySection>
        <LazySection id="model-lab">
          <ModelLabWrapper />
        </LazySection>
      </main>
      <FooterSection />
    </>
  );
}

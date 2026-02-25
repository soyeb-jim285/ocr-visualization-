import { EpochPrefetcher } from "@/components/EpochPrefetcher";
import { PixelViewSection } from "@/components/sections/PixelViewSection";
import { ConvolutionSection } from "@/components/sections/ConvolutionSection";
import { ActivationSection } from "@/components/sections/ActivationSection";
import { SecondConvSection } from "@/components/sections/SecondConvSection";
import { PoolingSection } from "@/components/sections/PoolingSection";
import { DeeperLayersSection } from "@/components/sections/DeeperLayersSection";
import { FullyConnectedSection } from "@/components/sections/FullyConnectedSection";
import { SoftmaxSection } from "@/components/sections/SoftmaxSection";
import { TrainingSection } from "@/components/sections/TrainingSection";
import { NeuronInspectorSection } from "@/components/sections/NeuronInspectorSection";
import { NeuronNetworkSection } from "@/components/sections/NeuronNetworkSection";
import { ScrollProgress } from "@/components/ui/ScrollProgress";
import { ViewToggle } from "@/components/ui/ViewToggle";
import { ScrollTracker } from "@/components/ui/ScrollTracker";
import { NetworkView3D } from "@/components/three/NetworkView3D";
import { LazySection } from "@/components/ui/LazySection";

export default function Home() {
  return (
    <>
      <EpochPrefetcher />
      <ScrollProgress />
      <ViewToggle />
      <ScrollTracker />
      <NetworkView3D />
      <main>
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
        <LazySection id="neuron-inspector">
          <NeuronInspectorSection />
        </LazySection>
      </main>
    </>
  );
}

import { PixelViewSection } from "@/components/sections/PixelViewSection";
import { ConvolutionSection } from "@/components/sections/ConvolutionSection";
import { ActivationSection } from "@/components/sections/ActivationSection";
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
      <ScrollProgress />
      <ViewToggle />
      <ScrollTracker />
      <NetworkView3D />
      <main>
        <NeuronNetworkSection />
        <LazySection>
          <PixelViewSection />
        </LazySection>
        <LazySection>
          <ConvolutionSection />
        </LazySection>
        <LazySection>
          <ActivationSection />
        </LazySection>
        <LazySection>
          <PoolingSection />
        </LazySection>
        <LazySection>
          <DeeperLayersSection />
        </LazySection>
        <LazySection>
          <FullyConnectedSection />
        </LazySection>
        <LazySection>
          <SoftmaxSection />
        </LazySection>
        <LazySection>
          <TrainingSection />
        </LazySection>
        <LazySection>
          <NeuronInspectorSection />
        </LazySection>
      </main>
    </>
  );
}

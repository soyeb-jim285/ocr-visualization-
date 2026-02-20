import { HeroSection } from "@/components/sections/HeroSection";
import { PixelViewSection } from "@/components/sections/PixelViewSection";
import { ConvolutionSection } from "@/components/sections/ConvolutionSection";
import { ActivationSection } from "@/components/sections/ActivationSection";
import { PoolingSection } from "@/components/sections/PoolingSection";
import { DeeperLayersSection } from "@/components/sections/DeeperLayersSection";
import { FullyConnectedSection } from "@/components/sections/FullyConnectedSection";
import { SoftmaxSection } from "@/components/sections/SoftmaxSection";
import { TrainingSection } from "@/components/sections/TrainingSection";
import { NeuronInspectorSection } from "@/components/sections/NeuronInspectorSection";
import { ScrollProgress } from "@/components/ui/ScrollProgress";
import { ViewToggle } from "@/components/ui/ViewToggle";
import { ScrollTracker } from "@/components/ui/ScrollTracker";
import { NetworkView3D } from "@/components/three/NetworkView3D";

export default function Home() {
  return (
    <>
      <ScrollProgress />
      <ViewToggle />
      <ScrollTracker />
      <NetworkView3D />
      <main>
        <HeroSection />
        <PixelViewSection />
        <ConvolutionSection />
        <ActivationSection />
        <PoolingSection />
        <DeeperLayersSection />
        <FullyConnectedSection />
        <SoftmaxSection />
        <TrainingSection />
        <NeuronInspectorSection />
      </main>
    </>
  );
}

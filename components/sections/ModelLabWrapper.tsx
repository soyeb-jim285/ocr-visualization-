"use client";

import dynamic from "next/dynamic";

const ModelLabSection = dynamic(
  () =>
    import("@/components/sections/ModelLabSection").then(
      (m) => m.ModelLabSection,
    ),
  { ssr: false },
);

export function ModelLabWrapper() {
  return <ModelLabSection />;
}

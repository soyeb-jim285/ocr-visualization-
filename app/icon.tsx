import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 32,
          height: 32,
          background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
          borderRadius: 6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: 18,
            height: 18,
            display: "flex",
            flexDirection: "column",
            gap: 2,
          }}
        >
          {/* Stylized neuron grid */}
          <div style={{ display: "flex", gap: 2 }}>
            <div style={{ width: 5, height: 5, background: "#fff", borderRadius: 1 }} />
            <div style={{ width: 5, height: 5, background: "rgba(255,255,255,0.6)", borderRadius: 1 }} />
            <div style={{ width: 5, height: 5, background: "#fff", borderRadius: 1 }} />
          </div>
          <div style={{ display: "flex", gap: 2 }}>
            <div style={{ width: 5, height: 5, background: "rgba(255,255,255,0.4)", borderRadius: 1 }} />
            <div style={{ width: 5, height: 5, background: "#fff", borderRadius: 1 }} />
            <div style={{ width: 5, height: 5, background: "rgba(255,255,255,0.6)", borderRadius: 1 }} />
          </div>
          <div style={{ display: "flex", gap: 2 }}>
            <div style={{ width: 5, height: 5, background: "#fff", borderRadius: 1 }} />
            <div style={{ width: 5, height: 5, background: "rgba(255,255,255,0.4)", borderRadius: 1 }} />
            <div style={{ width: 5, height: 5, background: "#fff", borderRadius: 1 }} />
          </div>
        </div>
      </div>
    ),
    { ...size }
  );
}

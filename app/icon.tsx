import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

// 3x3 specimen grid: lit cells trace a "V" in phosphor, one coral argmax pin
const CELLS = [1, 0, 1, 1, 0, 1, 0, 2, 0];

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 32,
          height: 32,
          background: "#06080b",
          border: "1px solid rgba(170,205,225,0.3)",
          borderRadius: 4,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div style={{ width: 20, display: "flex", flexWrap: "wrap", gap: 2 }}>
          {CELLS.map((c, i) => (
            <div
              key={i}
              style={{
                width: 6,
                height: 6,
                borderRadius: 1,
                background: c === 2 ? "#ff6b4a" : c === 1 ? "#8fe3ff" : "rgba(170,205,225,0.18)",
              }}
            />
          ))}
        </div>
      </div>
    ),
    { ...size }
  );
}

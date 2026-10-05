import { ImageResponse } from "next/og";

export const alt = "Neural Network X-Ray — Interactive CNN Visualization";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const COLS = [
  { n: 3, c: "#d7dee5" },
  { n: 5, c: "#7f8cff" },
  { n: 5, c: "#b08cff" },
  { n: 6, c: "#35d0e6" },
  { n: 5, c: "#4fe3a0" },
  { n: 4, c: "#ffb347" },
  { n: 3, c: "#ff6b4a" },
];

export default function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: "radial-gradient(ellipse 80% 70% at 50% 0%, #0f1a23, #06080b 70%)",
          fontFamily: "system-ui, sans-serif",
          padding: 72,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 20,
            letterSpacing: 3,
            color: "#8896a3",
            fontFamily: "monospace",
          }}
        >
          <span>NNXR</span>
          <span>PLATE 00 · 13 LAYERS · 146 CLASSES</span>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div
            style={{
              fontSize: 96,
              fontWeight: 300,
              color: "#e9eef2",
              lineHeight: 1,
              letterSpacing: -2,
              display: "flex",
            }}
          >
            Neural Network X-Ray
          </div>
          <div
            style={{
              fontSize: 28,
              color: "#b4c0ca",
              lineHeight: 1.45,
              marginTop: 28,
              maxWidth: 820,
              display: "flex",
            }}
          >
            Draw a character and watch every layer of a CNN read it, from raw pixels to a confident guess.
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", height: 1, background: "rgba(170,205,225,0.3)", marginBottom: 28 }} />
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", gap: 36, alignItems: "center" }}>
              {COLS.map((col, li) => (
                <div key={li} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {Array.from({ length: col.n }).map((_, ni) => (
                    <div
                      key={ni}
                      style={{
                        width: 14,
                        height: 14,
                        borderRadius: 7,
                        border: `1px solid ${col.c}`,
                        background: (ni + li) % 2 === 0 ? col.c : "transparent",
                        opacity: (ni + li) % 2 === 0 ? 0.85 : 0.5,
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
            <div
              style={{
                display: "flex",
                fontSize: 18,
                letterSpacing: 2,
                color: "#8fe3ff",
                fontFamily: "monospace",
              }}
            >
              RUNS IN YOUR BROWSER
            </div>
          </div>
        </div>
      </div>
    ),
    { ...size }
  );
}

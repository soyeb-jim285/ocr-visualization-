import { ImageResponse } from "next/og";

export const alt = "Neural Network X-Ray — Interactive CNN Visualization";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(145deg, #0a0a0f 0%, #13131a 50%, #1a1a2e 100%)",
          fontFamily: "system-ui, sans-serif",
          padding: 60,
        }}
      >
        {/* Network visualization hint — stylized layer nodes */}
        <div
          style={{
            display: "flex",
            gap: 32,
            alignItems: "center",
            marginBottom: 48,
          }}
        >
          {[4, 8, 12, 16, 12, 8, 6].map((n, li) => (
            <div
              key={li}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 4,
                alignItems: "center",
              }}
            >
              {Array.from({ length: Math.min(n, 6) }).map((_, ni) => (
                <div
                  key={ni}
                  style={{
                    width: 12,
                    height: 12,
                    borderRadius: 6,
                    background:
                      li === 0
                        ? "#06b6d4"
                        : li === 6
                          ? "#8b5cf6"
                          : `rgba(99, 102, 241, ${0.3 + (ni / 6) * 0.7})`,
                  }}
                />
              ))}
            </div>
          ))}
        </div>

        {/* Title */}
        <div
          style={{
            fontSize: 56,
            fontWeight: 700,
            color: "#e8e8ed",
            textAlign: "center",
            lineHeight: 1.2,
            marginBottom: 16,
          }}
        >
          Neural Network X-Ray
        </div>

        {/* Subtitle */}
        <div
          style={{
            fontSize: 24,
            color: "rgba(232, 232, 237, 0.5)",
            textAlign: "center",
            maxWidth: 800,
            lineHeight: 1.5,
          }}
        >
          Draw a character and watch every layer of a CNN process it in
          real time — from raw pixels to confident prediction.
        </div>

        {/* Tech pills */}
        <div
          style={{
            display: "flex",
            gap: 12,
            marginTop: 36,
          }}
        >
          {["Next.js", "ONNX Runtime", "React 19", "TypeScript"].map(
            (tech) => (
              <div
                key={tech}
                style={{
                  padding: "6px 16px",
                  borderRadius: 20,
                  border: "1px solid rgba(99, 102, 241, 0.3)",
                  color: "rgba(99, 102, 241, 0.8)",
                  fontSize: 16,
                }}
              >
                {tech}
              </div>
            )
          )}
        </div>
      </div>
    ),
    { ...size }
  );
}

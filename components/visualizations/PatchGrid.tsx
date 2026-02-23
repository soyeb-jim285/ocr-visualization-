"use client";

interface PatchGridProps {
  /** 2D array of values */
  data: number[][];
  /** Color function: value → [r, g, b] */
  colorFn: (val: number) => [number, number, number];
  /** Pixel size per cell */
  cellSize?: number;
  /** Show numeric values on cells */
  showValues?: boolean;
  /** Format function for displayed values */
  valueFormat?: (v: number) => string;
  /** Label below the grid */
  label?: string;
  /** Highlight border */
  highlight?: boolean;
  /** Click handler */
  onClick?: () => void;
}

/** Luminance-based contrast: returns white or black text */
function textColor(r: number, g: number, b: number): string {
  const lum = 0.299 * r + 0.587 * g + 0.114 * b;
  return lum > 140 ? "#000" : "#fff";
}

/**
 * Renders a small NxN matrix as a tight grid of colored cells.
 * No gaps between cells, no rounded corners — looks like an image patch.
 */
export function PatchGrid({
  data,
  colorFn,
  cellSize = 40,
  showValues = false,
  valueFormat = (v) => v.toFixed(2),
  label,
  highlight = false,
  onClick,
}: PatchGridProps) {
  const rows = data.length;
  const cols = data[0]?.length ?? 0;

  return (
    <div
      className={`flex flex-col items-center gap-1.5 ${onClick ? "cursor-pointer" : ""}`}
      onClick={onClick}
    >
      <div
        className={`overflow-hidden border ${
          highlight
            ? "border-accent-primary shadow-md shadow-accent-primary/20"
            : "border-border/60"
        }`}
        style={{ lineHeight: 0 }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
            gridTemplateRows: `repeat(${rows}, ${cellSize}px)`,
            gap: 0,
          }}
        >
          {data.flatMap((row, r) =>
            row.map((val, c) => {
              const [cr, cg, cb] = colorFn(val);
              const bg = `rgb(${cr}, ${cg}, ${cb})`;
              const fg = textColor(cr, cg, cb);
              return (
                <div
                  key={`${r}-${c}`}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: bg,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {showValues && (
                    <span
                      className="font-mono leading-none"
                      style={{
                        fontSize: Math.max(8, cellSize * 0.28),
                        color: fg,
                        opacity: 0.9,
                      }}
                    >
                      {valueFormat(val)}
                    </span>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
      {label && (
        <span className="text-xs text-foreground/40">{label}</span>
      )}
    </div>
  );
}

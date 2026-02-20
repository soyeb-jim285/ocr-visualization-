"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LAYOUTS = [
  { path: "/", label: "Main", short: "M" },
  { path: "/1", label: "Digital Rain", short: "1" },
  { path: "/2", label: "Oscilloscope", short: "2" },
  { path: "/3", label: "Topographic", short: "3" },
  { path: "/4", label: "Circuit Board", short: "4" },
  { path: "/5", label: "Cosmos", short: "5" },
  { path: "/6", label: "Terminal", short: "6" },
  { path: "/7", label: "Cathedral", short: "7" },
  { path: "/8", label: "Musical", short: "8" },
  { path: "/9", label: "Data Flow", short: "9" },
];

export function LayoutNav() {
  const pathname = usePathname();

  return (
    <nav className="fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 items-center gap-1 rounded-full border border-border bg-surface/90 px-2 py-1.5 backdrop-blur-md">
      {LAYOUTS.map((l) => {
        const active = pathname === l.path;
        return (
          <Link
            key={l.path}
            href={l.path}
            className={`flex h-8 min-w-8 items-center justify-center rounded-full px-2 text-xs font-medium transition-all ${
              active
                ? "bg-accent-primary text-white"
                : "text-foreground/50 hover:bg-surface-elevated hover:text-foreground/80"
            }`}
            title={l.label}
          >
            {l.short}
          </Link>
        );
      })}
    </nav>
  );
}

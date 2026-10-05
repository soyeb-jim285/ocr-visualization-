"use client"

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Toggle as TogglePrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

const toggleVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-[2px] border border-rule font-mono text-[11px] text-ink-2 hover:border-rule-strong hover:text-ink data-[state=on]:border-[var(--sig)] data-[state=on]:bg-[color-mix(in_oklab,var(--sig)_14%,transparent)] data-[state=on]:text-ink disabled:pointer-events-none disabled:opacity-40 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 [&_svg]:shrink-0 outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-phosphor transition-[color,border-color,background-color] duration-150 aria-invalid:border-destructive whitespace-nowrap pointer-coarse:min-h-11",
  {
    variants: {
      variant: {
        default: "bg-transparent",
        outline:
          "bg-transparent",
      },
      size: {
        default: "h-8 px-3 min-w-8",
        sm: "h-7 px-2 min-w-7",
        lg: "h-9 px-3 min-w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Toggle({
  className,
  variant,
  size,
  ...props
}: React.ComponentProps<typeof TogglePrimitive.Root> &
  VariantProps<typeof toggleVariants>) {
  return (
    <TogglePrimitive.Root
      data-slot="toggle"
      className={cn(toggleVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Toggle, toggleVariants }

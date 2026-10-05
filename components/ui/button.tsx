import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[3px] text-[13px] font-medium transition-[background-color,color,border-color,transform] duration-150 active:translate-y-px disabled:pointer-events-none disabled:opacity-40 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-phosphor aria-invalid:border-destructive",
  {
    variants: {
      variant: {
        default: "bg-ink text-bg hover:bg-phosphor",
        destructive: "bg-destructive text-bg hover:bg-destructive/85",
        outline:
          "border border-rule-strong bg-transparent text-ink-2 hover:border-ink-3 hover:text-ink",
        secondary:
          "border border-rule bg-bg-lift text-ink-2 hover:border-rule-strong hover:text-ink",
        ghost: "text-ink-2 hover:bg-bg-lift hover:text-ink",
        link: "font-mono text-[11px] tracking-[0.06em] text-ink-2 underline-offset-4 hover:text-phosphor hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        xs: "h-6 gap-1 rounded-[3px] px-2 text-xs has-[>svg]:px-1.5 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-8 rounded-[3px] gap-1.5 px-3 has-[>svg]:px-2.5",
        lg: "h-10 rounded-[3px] px-6 has-[>svg]:px-4",
        icon: "size-9",
        "icon-xs": "size-6 rounded-[3px] [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-8",
        "icon-lg": "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center justify-center rounded-[2px] border border-transparent px-2 py-0.5 font-mono text-[11px] font-medium tracking-[0.04em] w-fit whitespace-nowrap shrink-0 [&>svg]:size-3 gap-1 [&>svg]:pointer-events-none outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-phosphor aria-invalid:border-destructive transition-colors overflow-hidden",
  {
    variants: {
      variant: {
      default: "bg-ink text-bg [a&]:hover:bg-phosphor",
      secondary: "border-rule bg-bg-lift text-ink-2 [a&]:hover:text-ink",
      destructive: "bg-destructive text-bg [a&]:hover:bg-destructive/85",
      outline: "border-rule-strong text-ink-2 [a&]:hover:text-ink",
      ghost: "[a&]:hover:bg-bg-lift [a&]:hover:text-ink",
      link: "text-ink-2 underline-offset-4 [a&]:hover:text-phosphor [a&]:hover:underline",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "span"

  return (
    <Comp
      data-slot="badge"
      data-variant={variant}
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }

"use client"

import * as React from "react"
import { Tooltip as TooltipPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

function TooltipProvider({
  delayDuration = 0,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
  return (
    <TooltipPrimitive.Provider
      data-slot="tooltip-provider"
      delayDuration={delayDuration}
      {...props}
    />
  )
}

function Tooltip({
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Root>) {
  return <TooltipPrimitive.Root data-slot="tooltip" {...props} />
}

function TooltipTrigger({
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Trigger>) {
  return <TooltipPrimitive.Trigger data-slot="tooltip-trigger" {...props} />
}

function TooltipContent({
  className,
  sideOffset = 0,
  children,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content>) {
  return (
    <TooltipPrimitive.Portal forceMount>
      <TooltipPrimitive.Content
        data-slot="tooltip-content"
        sideOffset={sideOffset}
        forceMount
        className={cn(
          "z-50 w-fit rounded-md bg-foreground px-3 py-1.5 text-xs text-background text-balance origin-(--radix-tooltip-content-transform-origin)",
          "transition-all duration-200 ease-out",
          "data-[state=closed]:pointer-events-none data-[state=closed]:opacity-0 data-[state=closed]:scale-95",
          "data-[state=open]:opacity-100 data-[state=open]:scale-100",
          "data-[state=closed]:data-[side=bottom]:-translate-y-1 data-[state=closed]:data-[side=top]:translate-y-1 data-[state=closed]:data-[side=left]:translate-x-1 data-[state=closed]:data-[side=right]:-translate-x-1",
          className
        )}
        {...props}
      >
        {children}
      </TooltipPrimitive.Content>
    </TooltipPrimitive.Portal>
  )
}

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider }

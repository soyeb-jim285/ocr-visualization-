"use client";

/**
 * Previously used Lenis for smooth scrolling.
 * Removed to eliminate constant requestAnimationFrame loop overhead.
 * Native scroll with CSS scroll-behavior: smooth is used instead.
 */
export function SmoothScrollProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}

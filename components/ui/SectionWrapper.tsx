interface SectionWrapperProps {
  id: string;
  children: React.ReactNode;
  className?: string;
  fullHeight?: boolean;
}

export function SectionWrapper({
  id,
  children,
  className,
  fullHeight = true,
}: SectionWrapperProps) {
  return (
    <section
      id={id}
      className={`relative flex flex-col items-center justify-center px-3 py-10 sm:px-6 sm:py-16 md:px-8 ${
        fullHeight ? "min-h-screen" : ""
      } ${className ?? ""}`}
    >
      <div
        className="w-full border-t border-border/55 pt-8 sm:pt-10"
        style={{ maxWidth: "var(--content-max-width)" }}
      >
        {children}
      </div>
    </section>
  );
}

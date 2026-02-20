interface SectionHeaderProps {
  title: string;
  subtitle: string;
  step?: number;
}

export function SectionHeader({ title, subtitle, step }: SectionHeaderProps) {
  return (
    <div className="mb-12 text-center">
      {step !== undefined && (
        <span className="mb-3 inline-block rounded-full border border-accent-primary/30 bg-accent-primary/10 px-3 py-1 font-mono text-xs text-accent-primary">
          Step {step}
        </span>
      )}
      <h2
        className="mb-4 text-3xl font-bold tracking-tight sm:text-4xl md:text-5xl"
        style={{
          backgroundImage:
            "linear-gradient(135deg, var(--foreground), var(--foreground)/0.6)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}
      >
        {title}
      </h2>
      <p className="mx-auto max-w-2xl text-base text-foreground/50 sm:text-lg">
        {subtitle}
      </p>
    </div>
  );
}

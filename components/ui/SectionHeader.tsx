interface SectionHeaderProps {
  title: string;
  subtitle: string;
  step?: number;
}

export function SectionHeader({ title, subtitle, step }: SectionHeaderProps) {
  return (
    <div className="mb-8 text-center sm:mb-12">
      {step !== undefined && (
        <span className="mb-3 inline-block rounded-full border border-accent-primary/30 bg-accent-primary/10 px-3 py-1 font-mono text-xs text-accent-primary">
          Step {step}
        </span>
      )}
      <h2 className="mb-4 bg-gradient-to-br from-foreground to-foreground/60 bg-clip-text text-3xl font-bold tracking-tight text-transparent sm:text-4xl md:text-5xl">
        {title}
      </h2>
      <p className="mx-auto max-w-2xl text-base text-foreground/50 sm:text-lg">
        {subtitle}
      </p>
    </div>
  );
}

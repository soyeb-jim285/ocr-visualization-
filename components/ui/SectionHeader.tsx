interface SectionHeaderProps {
  title: string;
  subtitle: string;
  step?: number;
}

export function SectionHeader({ title, subtitle, step }: SectionHeaderProps) {
  return (
    <div className="mb-8 text-center sm:mb-10">
      {step !== undefined && (
        <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/45">
          Step {String(step).padStart(2, "0")}
        </p>
      )}
      <h2 className="mb-3 text-2xl font-semibold tracking-tight text-foreground sm:text-4xl md:text-[2.7rem]">
        {title}
      </h2>
      <p className="mx-auto max-w-3xl text-sm leading-relaxed text-foreground/58 sm:text-base md:text-[1.05rem]">
        {subtitle}
      </p>
    </div>
  );
}

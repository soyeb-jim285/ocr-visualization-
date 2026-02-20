interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  label?: string;
}

export function LoadingSpinner({ size = "md", label }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: "h-4 w-4 border",
    md: "h-6 w-6 border-2",
    lg: "h-10 w-10 border-2",
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={`animate-spin rounded-full border-accent-primary border-t-transparent ${sizeClasses[size]}`}
      />
      {label && <span className="text-sm text-foreground/40">{label}</span>}
    </div>
  );
}

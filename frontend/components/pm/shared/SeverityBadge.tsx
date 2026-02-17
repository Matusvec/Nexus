// ============================================
// SeverityBadge — Strategy §5
// ============================================
// Renders a colored pill: CRITICAL / HIGH / MEDIUM / LOW
// Light-mode colors per strategy specification.

import { cn } from "@/lib/utils";

export type Severity = "critical" | "high" | "medium" | "low";

const severityConfig: Record<
  Severity,
  { label: string; className: string; tooltip: string }
> = {
  critical: {
    label: "CRITICAL",
    className: "bg-red-100 text-red-700 border-red-200",
    tooltip: "Critical: Product is unusable for this use case.",
  },
  high: {
    label: "HIGH",
    className: "bg-orange-100 text-orange-700 border-orange-200",
    tooltip: "High: Major functionality is impaired.",
  },
  medium: {
    label: "MEDIUM",
    className: "bg-amber-100 text-amber-700 border-amber-200",
    tooltip: "Medium: Workaround exists but causes friction.",
  },
  low: {
    label: "LOW",
    className: "bg-green-100 text-green-700 border-green-200",
    tooltip: "Low: Minor inconvenience.",
  },
};

interface SeverityBadgeProps {
  severity: Severity;
  className?: string;
}

export function SeverityBadge({ severity, className }: SeverityBadgeProps) {
  const config = severityConfig[severity] ?? severityConfig.medium;

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-semibold tracking-[0.02em] uppercase",
        config.className,
        className
      )}
      title={config.tooltip}
      role="status"
    >
      {config.label}
    </span>
  );
}

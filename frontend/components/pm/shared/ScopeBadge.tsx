// ============================================
// ScopeBadge — Strategy §5
// ============================================
// Renders effort size pill: S / M / L / XL

import { cn } from "@/lib/utils";

export type ScopeSize = "XS" | "S" | "M" | "L" | "XL";

const scopeConfig: Record<ScopeSize, { className: string; label: string }> = {
  XS: {
    label: "XS",
    className: "bg-slate-100 text-slate-600 border-slate-200",
  },
  S: {
    label: "S",
    className: "bg-blue-100 text-blue-700 border-blue-200",
  },
  M: {
    label: "M",
    className: "bg-amber-100 text-amber-700 border-amber-200",
  },
  L: {
    label: "L",
    className: "bg-orange-100 text-orange-700 border-orange-200",
  },
  XL: {
    label: "XL",
    className: "bg-red-100 text-red-700 border-red-200",
  },
};

interface ScopeBadgeProps {
  scope: ScopeSize;
  className?: string;
}

export function ScopeBadge({ scope, className }: ScopeBadgeProps) {
  const config = scopeConfig[scope] ?? scopeConfig.M;

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-semibold tracking-[0.02em]",
        config.className,
        className
      )}
    >
      {config.label}
    </span>
  );
}

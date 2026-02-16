"use client";

import { cn } from "@/lib/utils";

type Severity = "critical" | "high" | "medium" | "low";

const severityClasses: Record<Severity, string> = {
  critical: "border-red-500/30 bg-red-500/10 text-red-400",
  high: "border-orange-500/30 bg-orange-500/10 text-orange-400",
  medium: "border-amber-500/30 bg-amber-500/10 text-amber-400",
  low: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
};

interface QuoteBlockProps {
  text: string;
  source: string;
  date?: string | null;
  severity: Severity;
  onClick?: () => void;
}

export default function QuoteBlock({
  text,
  source,
  date,
  severity,
  onClick,
}: QuoteBlockProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "w-full rounded-2xl border px-4 py-3 text-left transition hover:shadow-sm",
        severityClasses[severity]
      )}
    >
      <p className="text-sm leading-relaxed">"{text}"</p>
      <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
        <span>{source}</span>
        {date && <span>{date}</span>}
      </div>
    </button>
  );
}

"use client";

import { cn } from "@/lib/utils";

type Severity = "critical" | "high" | "medium" | "low";

const severityClasses: Record<Severity, string> = {
  critical: "border-red-300 bg-red-50 text-red-700",
  high: "border-orange-300 bg-orange-50 text-orange-700",
  medium: "border-amber-300 bg-amber-50 text-amber-700",
  low: "border-emerald-300 bg-emerald-50 text-emerald-700",
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

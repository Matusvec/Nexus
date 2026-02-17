"use client";

import { cn } from "@/lib/utils";
import { SeverityBadge, type Severity } from "./shared/SeverityBadge";

const severityClasses: Record<Severity, string> = {
  critical: "border-red-200 bg-red-50",
  high: "border-orange-200 bg-orange-50",
  medium: "border-amber-200 bg-amber-50",
  low: "border-green-200 bg-green-50",
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
  const Wrapper = onClick ? "button" : "div";
  return (
    <Wrapper
      type={onClick ? "button" : undefined}
      onClick={onClick}
      className={cn(
        "w-full rounded-2xl border px-4 py-3 text-left transition-shadow duration-200",
        severityClasses[severity],
        onClick && "cursor-pointer hover:shadow-sm"
      )}
    >
      <p className="text-sm leading-relaxed text-foreground">&ldquo;{text}&rdquo;</p>
      <div className="mt-3 flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          — {source}{date ? ` · ${date}` : ""}
        </span>
        <SeverityBadge severity={severity} />
      </div>
    </Wrapper>
  );
}

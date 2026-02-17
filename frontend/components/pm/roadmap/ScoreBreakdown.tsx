// ============================================
// ScoreBreakdown — Strategy §4.12
// ============================================
// Expandable formula visualization for roadmap priority scores.

"use client";

import type { ScoreBreakdown as ScoreBreakdownType } from "@/lib/pm/types";
import { cn } from "@/lib/utils";

interface ScoreBreakdownProps {
  breakdown: ScoreBreakdownType;
  className?: string;
}

function ValueBar({
  label,
  value,
  max,
  color,
  explanation,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
  explanation?: string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="flex items-center gap-3">
      <span className="w-20 text-xs text-muted-foreground text-right">
        {label}:
      </span>
      <span className="w-10 text-sm font-semibold text-foreground text-right">
        {value.toFixed(1)}
      </span>
      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-300", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      {explanation && (
        <span className="text-xs text-muted-foreground max-w-[180px] truncate">
          {explanation}
        </span>
      )}
    </div>
  );
}

export function ScoreBreakdown({ breakdown, className }: ScoreBreakdownProps) {
  return (
    <div className={cn("space-y-3 py-2", className)}>
      <p className="text-xs text-muted-foreground">
        Formula: {breakdown.formula || "(frequency × severity × weight) / effort"}
      </p>

      <div className="space-y-2">
        <ValueBar
          label="Frequency"
          value={breakdown.frequency.value}
          max={50}
          color="bg-primary"
          explanation={breakdown.frequency.explanation}
        />
        <ValueBar
          label="Severity"
          value={breakdown.severity.value}
          max={5}
          color="bg-orange-500"
        />
        <ValueBar
          label="Weight"
          value={breakdown.weight.value}
          max={3}
          color="bg-amber-500"
          explanation={breakdown.weight.reason}
        />
        <ValueBar
          label="Effort"
          value={breakdown.effort.value}
          max={16}
          color="bg-slate-400"
          explanation={breakdown.effort.scope}
        />
      </div>

      <div className="border-t border-border pt-2 flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Final:</span>
        <span className="text-lg font-bold text-foreground">
          {breakdown.final.toFixed(1)}
        </span>
        <span className="text-xs text-muted-foreground">
          = ({breakdown.frequency.value.toFixed(0)} × {breakdown.severity.value.toFixed(1)} ×{" "}
          {breakdown.weight.value.toFixed(1)}) / {breakdown.effort.value}
        </span>
      </div>
    </div>
  );
}

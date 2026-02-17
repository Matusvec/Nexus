// ============================================
// SkeletonTable — Strategy §5
// ============================================
// Shimmer skeleton for table loading states.

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface SkeletonTableProps {
  rows?: number;
  columns?: number;
  className?: string;
}

export function SkeletonTable({
  rows = 8,
  columns = 5,
  className,
}: SkeletonTableProps) {
  return (
    <div className={cn("rounded-2xl border border-border bg-card", className)}>
      {/* Header */}
      <div className="flex gap-4 border-b border-border px-5 py-3">
        {Array.from({ length: columns }).map((_, i) => (
          <Skeleton
            key={`h-${i}`}
            className="h-4 rounded-lg"
            style={{ width: `${60 + Math.random() * 40}%`, maxWidth: "120px" }}
          />
        ))}
      </div>
      {/* Rows */}
      {Array.from({ length: rows }).map((_, row) => (
        <div
          key={`r-${row}`}
          className="flex items-center gap-4 border-b border-border/50 px-5 py-3 last:border-b-0"
          style={{ height: "48px" }}
        >
          {Array.from({ length: columns }).map((_, col) => (
            <Skeleton
              key={`c-${row}-${col}`}
              className="h-3.5 rounded-lg"
              style={{
                width: col === 0 ? "50%" : `${40 + Math.random() * 30}%`,
                maxWidth: col === 0 ? "280px" : "120px",
              }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

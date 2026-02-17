// ============================================
// ClusterCard — Strategy §4.7
// ============================================
// Summary card with severity mini-bar and top quote.

import Link from "next/link";
import type { Cluster } from "@/lib/pm/types";
import { cn } from "@/lib/utils";

interface ClusterCardProps {
  cluster: Cluster;
  className?: string;
}

const severityColors: Record<string, string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-amber-500",
  low: "bg-green-500",
};

export function ClusterCard({ cluster, className }: ClusterCardProps) {
  // Severity distribution bar
  const distribution = cluster.severity_distribution;
  const total = distribution
    ? Object.values(distribution).reduce((a, b) => a + b, 0)
    : 0;

  // Get a representative quote if available
  const topQuote = cluster.top_quote;
  const topQuoteSource = cluster.top_quote_source;

  return (
    <Link
      href={`/pm/clusters/${cluster.id}`}
      className={cn(
        "group block rounded-2xl border border-border bg-card p-5 transition-shadow duration-200 hover:shadow-sm cursor-pointer",
        className
      )}
    >
      {/* Title + mention count */}
      <h3 className="text-base font-semibold text-foreground leading-tight">
        {cluster.label}
      </h3>
      <p className="mt-1 text-xs text-muted-foreground">
        {cluster.mention_count} mentions
      </p>

      {/* Severity distribution bar */}
      {distribution && total > 0 && (
        <div className="mt-3 flex h-2 w-full overflow-hidden rounded-full bg-muted">
          {(["critical", "high", "medium", "low"] as const).map((sev) => {
            const count = distribution[sev] ?? 0;
            if (count === 0) return null;
            return (
              <div
                key={sev}
                className={cn("h-full", severityColors[sev])}
                style={{ width: `${(count / total) * 100}%` }}
                title={`${sev}: ${count}`}
              />
            );
          })}
        </div>
      )}

      {/* Top quote preview */}
      {topQuote && (
        <div className="mt-3 rounded-xl bg-muted/50 px-3 py-2">
          <p className="text-xs leading-relaxed text-muted-foreground line-clamp-2">
            &ldquo;{topQuote}&rdquo;
          </p>
          {topQuoteSource && (
            <p className="mt-1 text-[11px] text-muted-foreground/70">
              — {topQuoteSource}
            </p>
          )}
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 text-xs font-medium text-primary group-hover:underline">
        View Details →
      </div>
    </Link>
  );
}

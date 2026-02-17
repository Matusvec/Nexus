import PageHeader from "@/components/pm/PageHeader";
import { StatusBadge } from "@/components/pm/shared/StatusBadge";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { RoadmapResponse } from "@/lib/pm/types";
import { Sparkles, ArrowRight } from "lucide-react";

export default async function ProposalsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = await searchParams;
  const statusFilter =
    typeof sp.status === "string" ? sp.status : undefined;

  const roadmap = await pmFetchSafe<RoadmapResponse>("/roadmap");
  let proposals = roadmap?.items ?? [];

  // Filter by status if specified
  if (statusFilter) {
    proposals = proposals.filter(
      (p) => p.proposal.status === statusFilter
    );
  }

  const allStatuses = (roadmap?.items ?? []).reduce(
    (acc, p) => {
      const s = p.proposal.status ?? "draft";
      acc[s] = (acc[s] ?? 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Proposals"
        description="Feature proposals generated from clusters, ready for review and approval."
      />

      {/* Status filter tabs */}
      {(roadmap?.items?.length ?? 0) > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <Link
            href="/pm/proposals"
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors duration-150 ${
              !statusFilter
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:bg-muted"
            }`}
          >
            All ({roadmap?.items?.length ?? 0})
          </Link>
          {Object.entries(allStatuses).map(([status, count]) => (
            <Link
              key={status}
              href={`/pm/proposals?status=${status}`}
              className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors duration-150 ${
                statusFilter === status
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:bg-muted"
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)} ({count})
            </Link>
          ))}
        </div>
      )}

      {proposals.length === 0 ? (
        <EmptyState
          icon={Sparkles}
          title={
            statusFilter ? "No proposals match this filter" : "No proposals yet"
          }
          description={
            statusFilter
              ? "Try a different status filter or view all."
              : "Generate proposals from a cluster to populate this list."
          }
          actionLabel={statusFilter ? "View All" : "View Clusters"}
          actionHref={statusFilter ? "/pm/proposals" : "/pm/clusters"}
        />
      ) : (
        <div className="space-y-3">
          {proposals.map((entry) => (
            <Link
              key={entry.proposal.id}
              href={`/pm/proposals/${entry.proposal.id}`}
              className="block rounded-2xl border border-border bg-card p-5 transition-colors duration-150 hover:bg-muted/30"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-semibold truncate">
                      {entry.proposal.title}
                    </h3>
                    <StatusBadge
                      status={entry.proposal.status ?? "draft"}
                    />
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                    {entry.proposal.description}
                  </p>
                </div>
                {entry.proposal.priority_score != null && (
                  <span className="shrink-0 rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary tabular-nums">
                    {entry.proposal.priority_score.toFixed(1)}
                  </span>
                )}
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                <span>Cluster: {entry.cluster_label}</span>
                <span>{entry.mention_count} mentions</span>
                {entry.proposal.effort && (
                  <span className="rounded-lg bg-muted px-1.5 py-0.5">
                    Effort: {entry.proposal.effort}
                  </span>
                )}
                {entry.proposal.impact && (
                  <span className="rounded-lg bg-muted px-1.5 py-0.5">
                    Impact: {entry.proposal.impact}
                  </span>
                )}
                <span className="ml-auto inline-flex items-center gap-1 text-primary font-medium">
                  View Details
                  <ArrowRight className="h-3 w-3" />
                </span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

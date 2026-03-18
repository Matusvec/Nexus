import PageHeader from "@/components/pm/PageHeader";
import { StatusBadge } from "@/components/pm/shared/StatusBadge";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { PaginatedResponse, Proposal } from "@/lib/pm/types";
import { Sparkles, ArrowRight } from "lucide-react";

export default async function ProposalsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = await searchParams;
  const statusFilter =
    typeof sp.status === "string" ? sp.status : undefined;
  const page = typeof sp.page === "string" ? parseInt(sp.page, 10) : 1;

  // Fetch from /proposals with server-side filtering
  const statusQuery = statusFilter ? `&status=${statusFilter}` : "";
  const proposals = await pmFetchSafe<PaginatedResponse<Proposal>>(
    `/proposals?page=${page}&per_page=20${statusQuery}`
  );

  // Also fetch unfiltered total for status counts
  const allProposals = statusFilter
    ? await pmFetchSafe<PaginatedResponse<Proposal>>("/proposals?page=1&per_page=100")
    : proposals;

  const items = proposals?.items ?? [];
  const totalItems = proposals?.total ?? 0;

  const allStatuses = (allProposals?.items ?? []).reduce(
    (acc, p) => {
      const s = p.status ?? "draft";
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
      {(allProposals?.total ?? 0) > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <Link
            href="/pm/proposals"
            className={`rounded-xl border px-3 py-1.5 text-xs font-medium transition-colors duration-150 ${
              !statusFilter
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:bg-muted"
            }`}
          >
            All ({allProposals?.total ?? 0})
          </Link>
          {Object.entries(allStatuses).map(([status, count]) => (
            <Link
              key={status}
              href={`/pm/proposals?status=${status}`}
              className={`rounded-xl border px-3 py-1.5 text-xs font-medium transition-colors duration-150 ${
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

      {items.length === 0 ? (
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
          {items.map((proposal, i) => (
            <Link
              key={proposal.id}
              href={`/pm/proposals/${proposal.id}`}
              className="block rounded-2xl border border-border bg-card p-5 opacity-0 animate-fade-in transition-colors duration-150 hover:bg-muted/30"
              style={{ animationDelay: `${i * 0.04}s` }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-semibold truncate">
                      {proposal.title}
                    </h3>
                    <StatusBadge
                      status={proposal.status ?? "draft"}
                    />
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                    {proposal.description}
                  </p>
                </div>
                {proposal.priority_score != null && (
                  <span className="shrink-0 rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary tabular-nums">
                    {proposal.priority_score.toFixed(1)}
                  </span>
                )}
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                {proposal.effort && (
                  <span className="rounded-xl bg-muted px-1.5 py-0.5">
                    Effort: {proposal.effort}
                  </span>
                )}
                {proposal.impact && (
                  <span className="rounded-xl bg-muted px-1.5 py-0.5">
                    Impact: {proposal.impact}
                  </span>
                )}
                <span className="ml-auto inline-flex items-center gap-1 text-primary font-medium">
                  View Details
                  <ArrowRight className="h-3 w-3" />
                </span>
              </div>
            </Link>
          ))}

          {/* Pagination */}
          {(proposals?.total_pages ?? 1) > 1 && (
            <div className="flex items-center justify-center gap-2 pt-4">
              {page > 1 && (
                <Link
                  href={`/pm/proposals?page=${page - 1}${statusFilter ? `&status=${statusFilter}` : ""}`}
                  className="rounded-xl border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground hover:bg-muted transition-colors duration-150"
                >
                  Previous
                </Link>
              )}
              <span className="text-xs text-muted-foreground">
                Page {page} of {proposals?.total_pages ?? 1}
              </span>
              {page < (proposals?.total_pages ?? 1) && (
                <Link
                  href={`/pm/proposals?page=${page + 1}${statusFilter ? `&status=${statusFilter}` : ""}`}
                  className="rounded-xl border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground hover:bg-muted transition-colors duration-150"
                >
                  Next
                </Link>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

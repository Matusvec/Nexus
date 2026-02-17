import PageHeader from "@/components/pm/PageHeader";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import { SeverityBadge } from "@/components/pm/shared/SeverityBadge";
import { StatusBadge } from "@/components/pm/shared/StatusBadge";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { ClusterDetail, ProblemMention } from "@/lib/pm/types";
import { Layers, ExternalLink } from "lucide-react";

export default async function ClusterDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const cluster = await pmFetchSafe<ClusterDetail>(`/clusters/${id}`);

  if (!cluster) {
    return (
      <div className="space-y-6">
        <PageHeader
          title="Cluster Detail"
          backLabel="Back to Clusters"
          backHref="/pm/clusters"
        />
        <EmptyState
          icon={Layers}
          title="Cluster not found"
          description="This cluster may have been deleted, or the backend is unavailable."
          actionLabel="Back to Clusters"
          actionHref="/pm/clusters"
        />
      </div>
    );
  }

  // Fetch member problems for severity data
  const memberProblems: (ProblemMention | null)[] = await Promise.all(
    cluster.members.slice(0, 50).map((m) =>
      pmFetchSafe<ProblemMention>(`/problems/${m.problem_id}`)
    )
  );
  const validProblems = memberProblems.filter(
    (p): p is ProblemMention => p !== null
  );

  // Compute severity distribution
  const severityDist: Record<string, number> = {};
  validProblems.forEach((p) => {
    severityDist[p.severity] = (severityDist[p.severity] ?? 0) + 1;
  });

  // Compute persona distribution
  const personaDist: Record<string, number> = {};
  validProblems.forEach((p) => {
    const key = p.persona ?? "Unknown";
    personaDist[key] = (personaDist[key] ?? 0) + 1;
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title={cluster.label}
        description={`${cluster.mention_count} mentions · threshold ${cluster.threshold}`}
        backLabel="Back to Clusters"
        backHref="/pm/clusters"
      />

      {/* Summary */}
      <div className="rounded-2xl border border-border bg-card p-6">
        <h2 className="text-base font-semibold">Summary</h2>
        <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
          {cluster.summary ?? "Summary has not been generated yet. Generate a proposal to enrich this cluster."}
        </p>
      </div>

      {/* Severity + Persona distribution */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Severity Distribution */}
        <div className="rounded-2xl border border-border bg-card p-5">
          <h3 className="text-sm font-semibold mb-3">Severity Distribution</h3>
          {Object.keys(severityDist).length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No severity data available.
            </p>
          ) : (
            <div className="space-y-2">
              {(["critical", "high", "medium", "low"] as const).map((sev) => {
                const count = severityDist[sev] ?? 0;
                if (count === 0) return null;
                const pct =
                  validProblems.length > 0
                    ? (count / validProblems.length) * 100
                    : 0;
                return (
                  <div key={sev} className="flex items-center gap-3">
                    <SeverityBadge severity={sev} />
                    <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          sev === "critical"
                            ? "bg-red-500"
                            : sev === "high"
                              ? "bg-orange-500"
                              : sev === "medium"
                                ? "bg-amber-500"
                                : "bg-green-500"
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-xs tabular-nums text-muted-foreground w-8 text-right">
                      {count}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Persona Breakdown */}
        <div className="rounded-2xl border border-border bg-card p-5">
          <h3 className="text-sm font-semibold mb-3">Persona Breakdown</h3>
          {Object.keys(personaDist).length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No persona data available.
            </p>
          ) : (
            <div className="space-y-2">
              {Object.entries(personaDist)
                .sort(([, a], [, b]) => b - a)
                .map(([persona, count]) => (
                  <div
                    key={persona}
                    className="flex items-center justify-between text-sm"
                  >
                    <span className="text-foreground">{persona}</span>
                    <span className="text-xs tabular-nums text-muted-foreground">
                      {count}
                    </span>
                  </div>
                ))}
            </div>
          )}
        </div>
      </div>

      {/* Members */}
      <div className="rounded-2xl border border-border bg-card p-6">
        <h2 className="text-base font-semibold">
          Members ({cluster.members.length})
        </h2>
        {cluster.members.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            No members in this cluster.
          </p>
        ) : (
          <div className="mt-4 space-y-2">
            {validProblems.map((problem) => {
              const member = cluster.members.find(
                (m) => m.problem_id === problem.id
              );
              return (
                <Link
                  key={problem.id}
                  href={`/pm/problems/${problem.id}`}
                  className="flex items-center justify-between gap-3 rounded-xl border border-border bg-card px-4 py-3 transition-colors duration-150 hover:bg-muted/30"
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">
                      {problem.problem_statement}
                    </p>
                    <p className="mt-0.5 text-xs text-muted-foreground line-clamp-1">
                      &ldquo;{problem.quote_text}&rdquo;
                    </p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <SeverityBadge severity={problem.severity} />
                    {member && (
                      <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] tabular-nums text-muted-foreground">
                        {(member.similarity * 100).toFixed(0)}%
                      </span>
                    )}
                    <ExternalLink className="h-3.5 w-3.5 text-muted-foreground" />
                  </div>
                </Link>
              );
            })}
            {/* Show remaining members not fetched */}
            {cluster.members.length > 50 && (
              <p className="text-xs text-muted-foreground text-center pt-2">
                Showing 50 of {cluster.members.length} members
              </p>
            )}
          </div>
        )}
      </div>

      {/* Proposals */}
      <div className="rounded-2xl border border-border bg-card p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold">Proposals</h2>
        </div>
        {cluster.proposals.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            No proposals generated yet for this cluster.
          </p>
        ) : (
          <div className="mt-4 space-y-3">
            {cluster.proposals.map((proposal) => (
              <Link
                key={proposal.id}
                href={`/pm/proposals/${proposal.id}`}
                className="block rounded-xl border border-border bg-card p-4 transition-colors duration-150 hover:bg-muted/30"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-medium">{proposal.title}</p>
                  {proposal.status && (
                    <StatusBadge status={proposal.status} />
                  )}
                </div>
                <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                  {proposal.description}
                </p>
                {proposal.priority_score != null && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Score:{" "}
                    <span className="font-medium text-foreground tabular-nums">
                      {proposal.priority_score.toFixed(1)}
                    </span>
                  </p>
                )}
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

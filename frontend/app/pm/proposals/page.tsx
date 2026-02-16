import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { RoadmapResponse } from "@/lib/pm/types";

export default async function ProposalsPage() {
  // Proposals are embedded inside roadmap response
  const roadmap = await pmFetchSafe<RoadmapResponse>("/roadmap");
  const proposals = roadmap?.items ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Proposals"
        description="Feature proposals generated from clusters, ready for review and approval."
      />

      {proposals.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
          No proposals yet. Generate proposals from a cluster to populate this list.
        </div>
      ) : (
        <div className="space-y-3">
          {proposals.map((entry) => (
            <div
              key={entry.proposal.id}
              className="rounded-2xl border border-border bg-card/70 p-5 transition hover:bg-muted/50"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-base font-semibold">{entry.proposal.title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                    {entry.proposal.description}
                  </p>
                </div>
                {entry.proposal.priority_score != null && (
                  <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                    Score {entry.proposal.priority_score.toFixed(1)}
                  </span>
                )}
              </div>
              <div className="mt-3 flex flex-wrap gap-3 text-xs text-muted-foreground">
                <span>Cluster: {entry.cluster_label}</span>
                <span>{entry.mention_count} mentions</span>
                {entry.proposal.effort && <span>Effort: {entry.proposal.effort}</span>}
                {entry.proposal.impact && <span>Impact: {entry.proposal.impact}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

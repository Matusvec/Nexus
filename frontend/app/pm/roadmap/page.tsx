import PageHeader from "@/components/pm/PageHeader";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { RoadmapResponse } from "@/lib/pm/types";
import { Map } from "lucide-react";

export default async function RoadmapPage() {
  const data = await pmFetchSafe<RoadmapResponse>("/roadmap");

  const items = data?.items ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Roadmap"
        description="Ranked proposals with explainable prioritization scores."
      />

      {items.length === 0 ? (
        <EmptyState
          icon={Map}
          title="Roadmap is empty"
          description="Generate proposals to populate the priority rankings."
          actionLabel="View Clusters"
          actionHref="/pm/clusters"
        />
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border bg-card">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-left text-[11px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
              <tr>
                <th className="px-4 py-3 w-12">Rank</th>
                <th className="px-4 py-3">Proposal</th>
                <th className="px-4 py-3">Cluster</th>
                <th className="px-4 py-3 text-right">Mentions</th>
                <th className="px-4 py-3 text-right">Score</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item, index) => (
                <tr
                  key={item.proposal.id}
                  className="border-t border-border transition-colors duration-100 hover:bg-muted/30"
                >
                  <td className="px-4 py-3 tabular-nums text-muted-foreground">
                    {index + 1}
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/pm/proposals/${item.proposal.id}`}
                      className="font-medium text-foreground hover:text-primary hover:underline"
                    >
                      {item.proposal.title}
                    </Link>
                    <p className="mt-0.5 text-xs text-muted-foreground line-clamp-1">
                      {item.proposal.description}
                    </p>
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/pm/clusters/${item.proposal.cluster_id}`}
                      className="text-muted-foreground hover:text-primary hover:underline"
                    >
                      {item.cluster_label}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {item.mention_count}
                  </td>
                  <td className="px-4 py-3 text-right">
                    {item.priority_score != null ? (
                      <span className="inline-flex items-center rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary tabular-nums">
                        {item.priority_score.toFixed(2)}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Footer */}
          <div className="border-t border-border bg-muted/30 px-4 py-2.5 text-xs text-muted-foreground">
            {items.length} proposal{items.length !== 1 ? "s" : ""} ranked
            {items[0]?.priority_score != null && (
              <span className="ml-2">
                · Top score:{" "}
                <span className="font-medium text-foreground">
                  {items[0].priority_score.toFixed(2)}
                </span>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

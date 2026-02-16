import PageHeader from "@/components/pm/PageHeader";
import { pmFetchSafe } from "@/lib/pm/api";
import type { RoadmapResponse } from "@/lib/pm/types";

export default async function RoadmapPage() {
  const data = await pmFetchSafe<RoadmapResponse>("/roadmap");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Roadmap"
        description="Ranked proposals with explainable prioritization."
      />

      {!data || data.items.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
          Roadmap is empty. Generate proposals to populate rankings.
        </div>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border bg-card/70">
          <table className="w-full text-sm">
            <thead className="bg-muted/70 text-left text-xs uppercase tracking-[0.2em] text-muted-foreground">
              <tr>
                <th className="px-4 py-3">Rank</th>
                <th className="px-4 py-3">Proposal</th>
                <th className="px-4 py-3">Cluster</th>
                <th className="px-4 py-3">Mentions</th>
                <th className="px-4 py-3">Score</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((item, index) => (
                <tr key={item.proposal.id} className="border-t border-border">
                  <td className="px-4 py-3">{index + 1}</td>
                  <td className="px-4 py-3 font-medium">
                    {item.proposal.title}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {item.cluster_label}
                  </td>
                  <td className="px-4 py-3">{item.mention_count}</td>
                  <td className="px-4 py-3">
                    {item.priority_score?.toFixed(2) ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

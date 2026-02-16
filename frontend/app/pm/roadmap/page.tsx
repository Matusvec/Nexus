import PageHeader from "@/components/pm/PageHeader";

interface RoadmapItem {
  proposal: {
    id: string;
    title: string;
    description: string | null;
    priority_score: number | null;
    impact: string | null;
    effort: string | null;
  };
  cluster_label: string;
  mention_count: number;
  priority_score: number | null;
}

interface RoadmapResponse {
  items: RoadmapItem[];
  total: number;
}

async function loadRoadmap(): Promise<RoadmapResponse | null> {
  try {
    const res = await fetch("/api/v1/roadmap", { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as RoadmapResponse;
  } catch {
    return null;
  }
}

export default async function RoadmapPage() {
  const data = await loadRoadmap();

  return (
    <div className="space-y-6">
      <PageHeader
        title="Roadmap"
        description="Ranked proposals with explainable prioritization."
      />

      {!data || data.items.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
          Roadmap is empty. Generate proposals to populate rankings.
        </div>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border bg-white/70">
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
                    {item.priority_score?.toFixed(2) ?? "--"}
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

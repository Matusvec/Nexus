import Link from "next/link";
import PageHeader from "@/components/pm/PageHeader";
import { pmFetchSafe } from "@/lib/pm/api";
import type { Cluster, PaginatedResponse } from "@/lib/pm/types";

export default async function ClustersPage() {
  const data = await pmFetchSafe<PaginatedResponse<Cluster>>(
    "/clusters?page=1&per_page=30",
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Clusters"
        description="Group problem mentions into themes and surface the biggest pain areas."
      />

      {!data || data.items.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
          No clusters yet. Run clustering after you embed problems.
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {data.items.map((cluster) => (
            <Link
              key={cluster.id}
              href={`/pm/clusters/${cluster.id}`}
              className="rounded-2xl border border-border bg-card/70 p-5 transition hover:-translate-y-1 hover:shadow-md"
            >
              <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                {cluster.mention_count} mentions
              </p>
              <h3 className="mt-3 text-lg font-semibold">{cluster.label}</h3>
              <p className="mt-2 text-sm text-muted-foreground line-clamp-2">
                {cluster.summary ?? "Summary pending. Generate a proposal to enrich."}
              </p>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

import PageHeader from "@/components/pm/PageHeader";
import RunPipelineButton from "@/components/pm/RunPipelineButton";
import { ClusterGrid } from "@/components/pm/clusters/ClusterGrid";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import { pmFetchSafe } from "@/lib/pm/api";
import type { Cluster, PaginatedResponse } from "@/lib/pm/types";
import { Layers } from "lucide-react";

export default async function ClustersPage() {
  const data = await pmFetchSafe<PaginatedResponse<Cluster>>(
    "/clusters?page=1&per_page=30",
  );

  const clusters = data?.items ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Clusters"
        description="Grouped problem themes ranked by frequency. Click a cluster to drill into its members and generate proposals."
        actions={<RunPipelineButton />}
      />

      {clusters.length === 0 ? (
        <EmptyState
          icon={Layers}
          title="No clusters yet"
          description="Run clustering after you embed problems to surface pain themes."
          actionLabel="View Problems"
          actionHref="/pm/problems"
        />
      ) : (
        <>
          <p className="text-sm text-muted-foreground">
            {clusters.length} cluster{clusters.length !== 1 ? "s" : ""} ·{" "}
            {clusters.reduce((sum, c) => sum + c.mention_count, 0)} total
            mentions
          </p>
          <ClusterGrid clusters={clusters} />
        </>
      )}
    </div>
  );
}

// ============================================
// ClusterGrid — Strategy §4.7
// ============================================
// Responsive card grid for clusters page.

import type { Cluster } from "@/lib/pm/types";
import { ClusterCard } from "./ClusterCard";

interface ClusterGridProps {
  clusters: Cluster[];
}

export function ClusterGrid({ clusters }: ClusterGridProps) {
  // Sort by mention count descending
  const sorted = [...clusters].sort(
    (a, b) => (b.mention_count ?? 0) - (a.mention_count ?? 0)
  );

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3">
      {sorted.map((cluster) => (
        <ClusterCard key={cluster.id} cluster={cluster} />
      ))}
    </div>
  );
}

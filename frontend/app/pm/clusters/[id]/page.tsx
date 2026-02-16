import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";

interface ClusterMember {
  id: string;
  problem_id: string;
  similarity: number;
}

interface Proposal {
  id: string;
  title: string;
  description: string | null;
}

interface ClusterDetail {
  id: string;
  label: string;
  summary: string | null;
  mention_count: number;
  members: ClusterMember[];
  proposals: Proposal[];
}

async function loadCluster(id: string): Promise<ClusterDetail | null> {
  try {
    const res = await fetch(`/api/v1/clusters/${id}`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as ClusterDetail;
  } catch {
    return null;
  }
}

export default async function ClusterDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const cluster = await loadCluster(params.id);

  if (!cluster) {
    return (
      <div className="space-y-6">
        <PageHeader title="Cluster Detail" />
        <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
          Cluster not found or backend unavailable.
        </div>
        <Link href="/pm/clusters" className="text-sm text-primary underline">
          Back to Clusters
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title={cluster.label}
        description={`${cluster.mention_count} mentions in this cluster.`}
      />

      <section className="rounded-2xl border border-border bg-white/70 p-6">
        <h2 className="text-lg font-semibold">Summary</h2>
        <p className="mt-3 text-sm text-muted-foreground">
          {cluster.summary ?? "Summary has not been generated yet."}
        </p>
      </section>

      <section className="rounded-2xl border border-border bg-white/70 p-6">
        <h2 className="text-lg font-semibold">Members</h2>
        {cluster.members.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            No members available.
          </p>
        ) : (
          <div className="mt-4 space-y-2 text-sm text-muted-foreground">
            {cluster.members.map((member) => (
              <div
                key={member.id}
                className="flex items-center justify-between rounded-xl border border-border bg-white px-4 py-3"
              >
                <span>Problem ID: {member.problem_id}</span>
                <span>Similarity {member.similarity.toFixed(2)}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-border bg-white/70 p-6">
        <h2 className="text-lg font-semibold">Proposals</h2>
        {cluster.proposals.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            No proposals created yet.
          </p>
        ) : (
          <div className="mt-4 space-y-3">
            {cluster.proposals.map((proposal) => (
              <div key={proposal.id} className="rounded-xl border border-border bg-white p-4">
                <p className="text-sm font-medium">{proposal.title}</p>
                <p className="mt-2 text-xs text-muted-foreground">
                  {proposal.description ?? "No description yet."}
                </p>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { ClusterDetail } from "@/lib/pm/types";

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
        <PageHeader title="Cluster Detail" />
        <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
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

      <section className="rounded-2xl border border-border bg-card/70 p-6">
        <h2 className="text-lg font-semibold">Summary</h2>
        <p className="mt-3 text-sm text-muted-foreground">
          {cluster.summary ?? "Summary has not been generated yet."}
        </p>
      </section>

      <section className="rounded-2xl border border-border bg-card/70 p-6">
        <h2 className="text-lg font-semibold">Members ({cluster.members.length})</h2>
        {cluster.members.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            No members available.
          </p>
        ) : (
          <div className="mt-4 space-y-2 text-sm">
            {cluster.members.map((member) => (
              <Link
                key={member.id}
                href={`/pm/problems/${member.problem_id}`}
                className="flex items-center justify-between rounded-xl border border-border bg-card px-4 py-3 transition hover:bg-muted/50"
              >
                <span className="text-muted-foreground font-mono text-xs">
                  {member.problem_id.slice(0, 8)}…
                </span>
                <span className="text-xs text-muted-foreground">
                  Similarity {(member.similarity * 100).toFixed(0)}%
                </span>
              </Link>
            ))}
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-border bg-card/70 p-6">
        <h2 className="text-lg font-semibold">Proposals</h2>
        {cluster.proposals.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            No proposals created yet.
          </p>
        ) : (
          <div className="mt-4 space-y-3">
            {cluster.proposals.map((proposal) => (
              <div key={proposal.id} className="rounded-xl border border-border bg-card p-4">
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

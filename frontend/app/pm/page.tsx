import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { PaginatedResponse, Evidence, ProblemMention, Cluster, RoadmapResponse } from "@/lib/pm/types";

export default async function PMDashboardPage() {
  const [evidence, problems, clusters, roadmap] = await Promise.all([
    pmFetchSafe<PaginatedResponse<Evidence>>("/evidence?page=1&per_page=1"),
    pmFetchSafe<PaginatedResponse<ProblemMention>>("/problems?page=1&per_page=1"),
    pmFetchSafe<PaginatedResponse<Cluster>>("/clusters?page=1&per_page=1"),
    pmFetchSafe<RoadmapResponse>("/roadmap"),
  ]);

  const cards = [
    { label: "Evidence", value: evidence?.total ?? 0, href: "/pm/evidence" },
    { label: "Problems", value: problems?.total ?? 0, href: "/pm/problems" },
    { label: "Clusters", value: clusters?.total ?? 0, href: "/pm/clusters" },
    { label: "Roadmap Items", value: roadmap?.total ?? 0, href: "/pm/roadmap" },
  ];

  return (
    <div className="space-y-8">
      <PageHeader
        title="Dashboard"
        description="Track pipeline health and jump into the next stage."
        actions={
          <Link
            href="/pm/evidence/upload"
            className="rounded-full bg-[hsl(var(--primary))] px-4 py-2 text-sm font-semibold text-primary-foreground shadow-sm"
          >
            Upload Evidence
          </Link>
        }
      />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => (
          <Link
            key={card.label}
            href={card.href}
            className="group rounded-2xl border border-border bg-card/70 p-4 transition hover:-translate-y-1 hover:shadow-md"
          >
            <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
              {card.label}
            </p>
            <p className="mt-3 text-3xl font-semibold">{card.value}</p>
            <p className="mt-2 text-xs text-muted-foreground">
              Open {card.label.toLowerCase()}
            </p>
          </Link>
        ))}
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-border bg-card/60 p-5">
          <h2 className="text-xl font-semibold">Next Best Actions</h2>
          <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
            <li>Upload new evidence and trigger extraction.</li>
            <li>Review high severity problems and tag them.</li>
            <li>Run clustering to surface the biggest pain themes.</li>
          </ul>
        </div>
        <div className="rounded-2xl border border-border bg-card/60 p-5">
          <h2 className="text-xl font-semibold">Pipeline Notes</h2>
          <p className="mt-3 text-sm text-muted-foreground">
            This workspace is tuned for fast reviews. Summaries stay high level,
            but every claim links back to a quote when you drill in.
          </p>
        </div>
      </section>
    </div>
  );
}

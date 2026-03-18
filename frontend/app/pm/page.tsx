import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";
import { pmFetchSafe, getLLMCalls } from "@/lib/pm/api";
import type {
  PaginatedResponse,
  Evidence,
  ProblemMention,
  ProblemStats,
  Cluster,
  RoadmapResponse,
} from "@/lib/pm/types";
import {
  FileText,
  AlertTriangle,
  Layers,
  Sparkles,
  Map,
  Upload,
  ArrowRight,
} from "lucide-react";

export default async function PMDashboardPage() {
  const [evidence, problems, problemStats, clusters, roadmap, calls] =
    await Promise.all([
      pmFetchSafe<PaginatedResponse<Evidence>>("/evidence?page=1&per_page=1"),
      pmFetchSafe<PaginatedResponse<ProblemMention>>(
        "/problems?page=1&per_page=1"
      ),
      pmFetchSafe<ProblemStats>("/problems/stats"),
      pmFetchSafe<PaginatedResponse<Cluster>>("/clusters?page=1&per_page=1"),
      pmFetchSafe<RoadmapResponse>("/roadmap"),
      pmFetchSafe<Record<string, unknown>[]>("/llm/calls"),
    ]);

  const evCount = evidence?.total ?? 0;
  const prCount = problems?.total ?? 0;
  const clCount = clusters?.total ?? 0;
  const proposalCount = roadmap?.items?.length ?? 0;
  const roadmapCount = roadmap?.total ?? 0;

  const isEmpty = evCount === 0;

  const cards = [
    {
      label: "Evidence",
      value: evCount,
      sub: "source documents",
      href: "/pm/evidence",
      icon: FileText,
    },
    {
      label: "Problems",
      value: prCount,
      sub: problemStats
        ? `${problemStats.by_severity?.critical ?? 0} critical`
        : "extracted mentions",
      href: "/pm/problems",
      icon: AlertTriangle,
    },
    {
      label: "Clusters",
      value: clCount,
      sub: "pain themes",
      href: "/pm/clusters",
      icon: Layers,
    },
    {
      label: "Proposals",
      value: proposalCount,
      sub: "feature specs",
      href: "/pm/proposals",
      icon: Sparkles,
    },
    {
      label: "Roadmap",
      value: roadmapCount,
      sub: "ranked items",
      href: "/pm/roadmap",
      icon: Map,
    },
  ];

  // Compute next actions based on pipeline state
  const nextActions: { emoji: string; text: string; href: string; label: string }[] =
    [];
  if (evCount === 0) {
    nextActions.push({
      emoji: "🔵",
      text: "Upload your first piece of evidence",
      href: "/pm/evidence/upload",
      label: "Upload Evidence →",
    });
  } else if (prCount === 0) {
    nextActions.push({
      emoji: "🟡",
      text: "Extract problems from uploaded evidence",
      href: "/pm/evidence",
      label: "View Evidence →",
    });
  }
  if (prCount > 0 && clCount === 0) {
    nextActions.push({
      emoji: "🟡",
      text: "Run clustering to surface pain themes",
      href: "/pm/clusters",
      label: "Run Clustering →",
    });
  }
  if (clCount > 0 && proposalCount === 0) {
    nextActions.push({
      emoji: "🟡",
      text: "Generate proposals from clusters",
      href: "/pm/clusters",
      label: "View Clusters →",
    });
  }
  if (evCount > 0) {
    nextActions.push({
      emoji: "🔵",
      text: "Upload more evidence for richer signal",
      href: "/pm/evidence/upload",
      label: "Upload Evidence →",
    });
  }

  // Recent jobs from LLM calls
  const recentJobs = Array.isArray(calls) ? calls.slice(0, 10) : [];

  return (
    <div className="space-y-8">
      <PageHeader
        title="Dashboard"
        description="Track pipeline health and jump into the next stage."
        actions={
          <Link
            href="/pm/evidence/upload"
            className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/90 active:scale-[0.98]"
          >
            <Upload className="h-4 w-4" strokeWidth={1.75} />
            Upload Evidence
          </Link>
        }
      />

      {/* First-use empty state */}
      {isEmpty ? (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-border bg-card py-16 px-8 text-center">
          <FileText
            className="h-8 w-8 text-muted-foreground mb-4"
            strokeWidth={1.5}
          />
          <h2 className="text-xl font-semibold">
            Start your product discovery pipeline
          </h2>
          <p className="mt-2 max-w-md text-sm text-muted-foreground">
            Upload a customer interview, support ticket, or sales note. Nexus
            will extract problems, find patterns, and generate feature
            proposals.
          </p>
          <Link
            href="/pm/evidence/upload"
            className="mt-6 inline-flex items-center gap-1.5 rounded-xl bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/90 active:scale-[0.98]"
          >
            Upload Your First Evidence →
          </Link>
        </div>
      ) : (
        <>
          {/* Stat cards */}
          <section className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-5">
            {cards.map((card, i) => (
              <Link
                key={card.label}
                href={card.href}
                className="group rounded-2xl border border-border bg-card p-5 opacity-0 animate-fade-scale-in transition-shadow duration-200 hover:shadow-sm"
                style={{ animationDelay: `${i * 0.06}s` }}
              >
                <div className="flex items-center gap-2 text-muted-foreground">
                  <card.icon className="h-4 w-4" strokeWidth={1.75} />
                  <span className="text-[11px] font-medium uppercase tracking-[0.1em]">
                    {card.label}
                  </span>
                </div>
                <p className="mt-3 text-3xl font-semibold tracking-tight">
                  {card.value}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">{card.sub}</p>
              </Link>
            ))}
          </section>

          {/* Next Actions + Recent Jobs */}
          <section className="grid gap-6 lg:grid-cols-2">
            {/* Next Actions Panel */}
            <div className="rounded-2xl border border-border bg-card p-6">
              <h2 className="text-base font-semibold">Next Actions</h2>
              <div className="mt-4 space-y-3">
                {nextActions.map((action, i) => (
                  <Link
                    key={i}
                    href={action.href}
                    className="flex items-center justify-between rounded-xl border border-border p-3 transition-colors duration-150 hover:bg-muted/50"
                  >
                    <span className="text-sm text-foreground">
                      {action.emoji} {action.text}
                    </span>
                    <span className="text-xs font-medium text-primary flex items-center gap-1">
                      {action.label}
                      <ArrowRight className="h-3.5 w-3.5" />
                    </span>
                  </Link>
                ))}
              </div>
            </div>

            {/* Recent Jobs */}
            <div className="rounded-2xl border border-border bg-card p-6">
              <h2 className="text-base font-semibold">Recent Jobs</h2>
              {recentJobs.length === 0 ? (
                <p className="mt-4 text-sm text-muted-foreground">
                  No jobs recorded yet. Jobs will appear after running
                  extraction, clustering, or proposal generation.
                </p>
              ) : (
                <div className="mt-4 space-y-2">
                  {recentJobs.map((job, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between rounded-xl border border-border/50 px-3 py-2 text-sm"
                    >
                      <span className="font-medium text-foreground">
                        {(job as Record<string, unknown>).job_type as string ??
                          (job as Record<string, unknown>).operation as string ??
                          "Job"}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {(job as Record<string, unknown>).duration_ms
                          ? `${((job as Record<string, unknown>).duration_ms as number / 1000).toFixed(1)}s`
                          : "—"}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

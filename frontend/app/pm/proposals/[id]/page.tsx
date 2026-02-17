import PageHeader from "@/components/pm/PageHeader";
import { StatusBadge } from "@/components/pm/shared/StatusBadge";
import { ScopeBadge } from "@/components/pm/shared/ScopeBadge";
import { SeverityBadge } from "@/components/pm/shared/SeverityBadge";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { ProposalDetail } from "@/lib/pm/types";
import {
  Sparkles,
  Target,
  AlertTriangle,
  CheckCircle2,
  Quote,
  ListChecks,
} from "lucide-react";

export default async function ProposalDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const proposal = await pmFetchSafe<ProposalDetail>(
    `/feature_proposals/${id}`
  );

  if (!proposal) {
    return (
      <div className="space-y-6">
        <PageHeader
          title="Proposal Detail"
          backLabel="Back to Proposals"
          backHref="/pm/proposals"
        />
        <EmptyState
          icon={Sparkles}
          title="Proposal not found"
          description="This proposal may not exist yet, or the backend is unavailable."
          actionLabel="Back to Proposals"
          actionHref="/pm/proposals"
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title={proposal.title}
        backLabel="Back to Proposals"
        backHref="/pm/proposals"
        actions={
          <div className="flex items-center gap-2">
            <StatusBadge status={proposal.status ?? "draft"} />
            <ScopeBadge scope={proposal.scope_estimate} />
          </div>
        }
      />

      {/* User Story */}
      {proposal.user_story && (
        <div className="rounded-2xl border border-border bg-card p-6">
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <Target className="h-4 w-4 text-primary" strokeWidth={1.75} />
            User Story
          </h2>
          <p className="mt-3 text-sm text-muted-foreground leading-relaxed italic">
            {proposal.user_story}
          </p>
        </div>
      )}

      {/* JTBD Framing */}
      {proposal.jtbd_framing && (
        <div className="rounded-2xl border border-border bg-card p-6">
          <h2 className="text-base font-semibold">Jobs-to-be-Done</h2>
          <p className="mt-3 text-sm text-muted-foreground leading-relaxed">
            {proposal.jtbd_framing}
          </p>
        </div>
      )}

      {/* Description + Rationale */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-border bg-card p-6">
          <h2 className="text-base font-semibold">Description</h2>
          <p className="mt-3 text-sm text-muted-foreground leading-relaxed">
            {proposal.description}
          </p>
        </div>
        <div className="rounded-2xl border border-border bg-card p-6">
          <h2 className="text-base font-semibold">Rationale</h2>
          <p className="mt-3 text-sm text-muted-foreground leading-relaxed">
            {proposal.rationale}
          </p>
        </div>
      </div>

      {/* Success Metrics */}
      {proposal.success_metrics.length > 0 && (
        <div className="rounded-2xl border border-border bg-card p-6">
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <CheckCircle2
              className="h-4 w-4 text-green-600"
              strokeWidth={1.75}
            />
            Success Metrics ({proposal.success_metrics.length})
          </h2>
          <div className="mt-4 space-y-3">
            {proposal.success_metrics.map((m, i) => (
              <div
                key={i}
                className="rounded-xl border border-border bg-muted/30 p-4"
              >
                <p className="text-sm font-medium">{m.metric}</p>
                <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
                  <span>
                    Target:{" "}
                    <span className="font-medium text-foreground">
                      {m.target}
                    </span>
                  </span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  {m.reasoning}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risks */}
      {proposal.risks.length > 0 && (
        <div className="rounded-2xl border border-border bg-card p-6">
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <AlertTriangle
              className="h-4 w-4 text-amber-600"
              strokeWidth={1.75}
            />
            Risks ({proposal.risks.length})
          </h2>
          <div className="mt-4 space-y-3">
            {proposal.risks.map((r, i) => (
              <div
                key={i}
                className="rounded-xl border border-border bg-muted/30 p-4"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-medium">{r.risk}</p>
                  <SeverityBadge severity={r.severity} />
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  <span className="font-medium">Mitigation:</span>{" "}
                  {r.mitigation}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Edge Cases */}
      {proposal.edge_cases.length > 0 && (
        <div className="rounded-2xl border border-border bg-card p-6">
          <h2 className="text-base font-semibold">
            Edge Cases ({proposal.edge_cases.length})
          </h2>
          <ul className="mt-3 space-y-2">
            {proposal.edge_cases.map((ec, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-sm text-muted-foreground"
              >
                <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground/40" />
                {ec}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Citations */}
      {proposal.citations.length > 0 && (
        <div className="rounded-2xl border border-border bg-card p-6">
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <Quote className="h-4 w-4 text-muted-foreground" strokeWidth={1.75} />
            Citations ({proposal.citations.length})
          </h2>
          <div className="mt-4 space-y-3">
            {proposal.citations.map((c) => (
              <div
                key={c.id}
                className="rounded-xl border-l-2 border-l-primary/40 bg-muted/30 p-4"
              >
                <p className="text-sm italic text-muted-foreground">
                  &ldquo;{c.quote_text}&rdquo;
                </p>
                <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                  <span>From: {c.evidence_title}</span>
                  <Link
                    href={`/pm/problems/${c.problem_id}`}
                    className="text-primary hover:underline"
                  >
                    View Problem →
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cluster link + Tasks status */}
      <div className="grid gap-4 sm:grid-cols-2">
        <Link
          href={`/pm/clusters/${proposal.cluster_id}`}
          className="rounded-2xl border border-border bg-card p-5 transition-colors duration-150 hover:bg-muted/30"
        >
          <p className="text-[11px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
            Source Cluster
          </p>
          <p className="mt-2 text-sm font-medium">
            {proposal.cluster?.label ?? proposal.cluster_id}
          </p>
        </Link>
        <div className="rounded-2xl border border-border bg-card p-5">
          <p className="text-[11px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
            Tasks
          </p>
          <div className="mt-2 flex items-center gap-2">
            <ListChecks
              className="h-4 w-4 text-muted-foreground"
              strokeWidth={1.75}
            />
            {proposal.tasks_generated ? (
              <Link
                href={`/pm/tasks?proposal=${proposal.id}`}
                className="text-sm font-medium text-primary hover:underline"
              >
                View Task Tree →
              </Link>
            ) : (
              <span className="text-sm text-muted-foreground">
                Tasks not generated yet
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Metadata footer */}
      <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
        <span>Version {proposal.version}</span>
        {proposal.created_at && (
          <span>
            Created {new Date(proposal.created_at).toLocaleDateString()}
          </span>
        )}
        {proposal.updated_at && (
          <span>
            Updated {new Date(proposal.updated_at).toLocaleDateString()}
          </span>
        )}
      </div>
    </div>
  );
}

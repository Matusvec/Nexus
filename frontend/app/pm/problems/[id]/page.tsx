import PageHeader from "@/components/pm/PageHeader";
import QuoteBlock from "@/components/pm/QuoteBlock";
import { SeverityBadge } from "@/components/pm/shared/SeverityBadge";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { ProblemMention, SimilarProblem } from "@/lib/pm/types";
import { AlertTriangle, ExternalLink } from "lucide-react";

export default async function ProblemDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const problem = await pmFetchSafe<ProblemMention>(`/problems/${id}`);

  let similar: SimilarProblem[] = [];
  if (problem) {
    const qs = new URLSearchParams({
      text: problem.problem_statement,
      limit: "5",
      min_score: "0.4",
    });
    const resp = await pmFetchSafe<{ results: SimilarProblem[] }>(
      `/problems/similar?${qs}`,
    );
    similar = (resp?.results ?? []).filter((s) => s.problem.id !== id);
  }

  if (!problem) {
    return (
      <div className="space-y-6">
        <PageHeader
          title="Problem Detail"
          backLabel="Back to Problems"
          backHref="/pm/problems"
        />
        <EmptyState
          icon={AlertTriangle}
          title="Problem not found"
          description="This problem may have been deleted, or the backend is unavailable."
          actionLabel="Back to Problems"
          actionHref="/pm/problems"
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title={problem.problem_statement}
        backLabel="Back to Problems"
        backHref="/pm/problems"
      />

      {/* Metadata bar */}
      <div className="flex flex-wrap items-center gap-3">
        <SeverityBadge severity={problem.severity} />
        {problem.persona && (
          <span className="rounded-lg bg-muted px-2 py-0.5 text-xs text-muted-foreground">
            {problem.persona}
          </span>
        )}
        {problem.segment && (
          <span className="rounded-lg bg-muted px-2 py-0.5 text-xs text-muted-foreground">
            {problem.segment}
          </span>
        )}
        {problem.tags.map((tag) => (
          <span
            key={tag}
            className="rounded-lg bg-muted px-2 py-0.5 text-xs text-muted-foreground"
          >
            {tag}
          </span>
        ))}
        <Link
          href={`/pm/evidence/${problem.evidence_id}`}
          className="ml-auto inline-flex items-center gap-1 text-xs text-primary hover:underline"
        >
          View Source Evidence
          <ExternalLink className="h-3 w-3" />
        </Link>
      </div>

      {/* Quote */}
      <QuoteBlock
        text={problem.quote_text}
        source={`Evidence ${problem.evidence_id.slice(0, 8)}…`}
        severity={problem.severity}
      />

      {/* Similar Problems */}
      <section className="rounded-2xl border border-border bg-card p-6">
        <h2 className="text-base font-semibold">Similar Problems</h2>
        {similar.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            No similar problems found. Embed problems first to enable
            similarity search.
          </p>
        ) : (
          <div className="mt-4 space-y-3">
            {similar.map((item) => (
              <Link
                key={item.problem.id}
                href={`/pm/problems/${item.problem.id}`}
                className="flex items-center justify-between gap-3 rounded-xl border border-border bg-card p-4 transition-colors duration-150 hover:bg-muted/30"
              >
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">
                    {item.problem.problem_statement}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground line-clamp-1">
                    &ldquo;{item.problem.quote_text}&rdquo;
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <SeverityBadge severity={item.problem.severity} />
                  <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary tabular-nums">
                    {(item.score * 100).toFixed(0)}%
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

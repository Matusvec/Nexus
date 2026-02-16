import PageHeader from "@/components/pm/PageHeader";
import QuoteBlock from "@/components/pm/QuoteBlock";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { ProblemMention, SimilarProblem } from "@/lib/pm/types";

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
    similar = resp?.results ?? [];
  }

  if (!problem) {
    return (
      <div className="space-y-6">
        <PageHeader title="Problem Detail" />
        <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
          Problem not found or backend unavailable.
        </div>
        <Link href="/pm/problems" className="text-sm text-primary underline">
          Back to Problems
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Problem Detail"
        description={`${problem.persona ?? "Unknown persona"} · ${problem.segment ?? "Unknown segment"} · ${problem.severity}`}
      />

      <QuoteBlock
        text={problem.quote_text}
        source={problem.problem_statement}
        severity={problem.severity}
      />

      <section className="rounded-2xl border border-border bg-card/70 p-6">
        <h2 className="text-lg font-semibold">Similar problems</h2>
        {similar.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            No similar problems found yet. Embed problems first to enable similarity search.
          </p>
        ) : (
          <div className="mt-4 space-y-3">
            {similar.map((item) => (
              <Link
                key={item.problem.id}
                href={`/pm/problems/${item.problem.id}`}
                className="block rounded-xl border border-border bg-card p-4 transition hover:bg-muted/50"
              >
                <p className="text-sm font-medium">{item.problem.problem_statement}</p>
                <p className="mt-1 text-xs text-muted-foreground line-clamp-1">
                  &ldquo;{item.problem.quote_text}&rdquo;
                </p>
                <p className="mt-2 text-xs text-muted-foreground">
                  Similarity: {(item.score * 100).toFixed(0)}%
                </p>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

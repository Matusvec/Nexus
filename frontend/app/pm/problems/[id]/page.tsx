import PageHeader from "@/components/pm/PageHeader";
import QuoteBlock from "@/components/pm/QuoteBlock";
import Link from "next/link";

interface ProblemDetail {
  id: string;
  problem_statement: string;
  severity: "critical" | "high" | "medium" | "low";
  quote_text: string;
  persona: string | null;
  segment: string | null;
}

interface SimilarProblemsResponse {
  results: { problem: ProblemDetail; score: number }[];
}

async function loadProblem(id: string): Promise<ProblemDetail | null> {
  try {
    const res = await fetch(`/api/v1/problems/${id}`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as ProblemDetail;
  } catch {
    return null;
  }
}

async function loadSimilar(statement: string): Promise<SimilarProblemsResponse | null> {
  try {
    const params = new URLSearchParams({
      text: statement,
      limit: "5",
      min_score: "0.4",
    });
    const res = await fetch(`/api/v1/problems/similar?${params}`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as SimilarProblemsResponse;
  } catch {
    return null;
  }
}

export default async function ProblemDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const problem = await loadProblem(params.id);
  const similar = problem ? await loadSimilar(problem.problem_statement) : null;

  if (!problem) {
    return (
      <div className="space-y-6">
        <PageHeader title="Problem Detail" />
        <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
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
        description={`${problem.persona ?? "Unknown persona"} • ${problem.segment ?? "Unknown segment"} • ${problem.severity}`}
      />

      <QuoteBlock
        text={problem.quote_text}
        source={problem.problem_statement}
        severity={problem.severity}
      />

      <section className="rounded-2xl border border-border bg-white/70 p-6">
        <h2 className="text-lg font-semibold">Similar problems</h2>
        {!similar || similar.results.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            No similar problems found yet.
          </p>
        ) : (
          <div className="mt-4 space-y-3">
            {similar.results.map((item) => (
              <div key={item.problem.id} className="rounded-xl border border-border bg-white p-4">
                <p className="text-sm font-medium">{item.problem.problem_statement}</p>
                <p className="mt-2 text-xs text-muted-foreground">
                  Similarity score: {item.score.toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";

interface ProblemMention {
  id: string;
  problem_statement: string;
  severity: string;
  persona: string | null;
  tags: string[];
  quote_text: string;
}

interface ProblemListResponse {
  items: ProblemMention[];
  total: number;
}

interface ProblemStats {
  total: number;
  by_severity: Record<string, number>;
}

async function loadProblems(): Promise<ProblemListResponse | null> {
  try {
    const res = await fetch("/api/v1/problems?page=1&per_page=20", {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as ProblemListResponse;
  } catch {
    return null;
  }
}

async function loadStats(): Promise<ProblemStats | null> {
  try {
    const res = await fetch("/api/v1/problems/stats", { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as ProblemStats;
  } catch {
    return null;
  }
}

export default async function ProblemsPage() {
  const [data, stats] = await Promise.all([loadProblems(), loadStats()]);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Problems"
        description="Scan extracted problem mentions, filter by severity, and jump to similar issues."
      />

      <section className="rounded-2xl border border-border bg-white/70 p-4">
        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
          Severity distribution
        </p>
        <div className="mt-3 flex flex-wrap gap-2 text-sm">
          {Object.entries(stats?.by_severity ?? {}).map(([severity, count]) => (
            <div
              key={severity}
              className="rounded-full border border-border bg-white px-3 py-1"
            >
              {severity}: {count}
            </div>
          ))}
        </div>
      </section>

      {!data || data.items.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
          No problem mentions yet. Extract problems from evidence to populate this table.
        </div>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border bg-white/70">
          <table className="w-full text-sm">
            <thead className="bg-muted/70 text-left text-xs uppercase tracking-[0.2em] text-muted-foreground">
              <tr>
                <th className="px-4 py-3">Problem</th>
                <th className="px-4 py-3">Severity</th>
                <th className="px-4 py-3">Persona</th>
                <th className="px-4 py-3">Tags</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((item) => (
                <tr key={item.id} className="border-t border-border">
                  <td className="px-4 py-3 font-medium">
                    <Link href={`/pm/problems/${item.id}`} className="hover:underline">
                      {item.problem_statement}
                    </Link>
                    <p className="mt-1 text-xs text-muted-foreground line-clamp-1">
                      {item.quote_text}
                    </p>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {item.severity}
                  </td>
                  <td className="px-4 py-3">{item.persona ?? "--"}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {item.tags.length ? item.tags.join(", ") : "--"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { ProblemMention, ProblemStats, PaginatedResponse } from "@/lib/pm/types";

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-500/20 text-red-400 border-red-500/30",
  high: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  medium: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  low: "bg-green-500/20 text-green-400 border-green-500/30",
};

export default async function ProblemsPage() {
  const [data, stats] = await Promise.all([
    pmFetchSafe<PaginatedResponse<ProblemMention>>("/problems?page=1&per_page=20"),
    pmFetchSafe<ProblemStats>("/problems/stats"),
  ]);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Problems"
        description="Scan extracted problem mentions, filter by severity, and jump to similar issues."
      />

      {/* Severity distribution */}
      <section className="rounded-2xl border border-border bg-card/70 p-4">
        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
          Severity distribution
        </p>
        <div className="mt-3 flex flex-wrap gap-2 text-sm">
          {stats &&
            Object.entries(stats.by_severity).map(([severity, count]) => (
              <div
                key={severity}
                className={`rounded-full border px-3 py-1 font-medium ${SEVERITY_COLORS[severity] ?? "bg-muted text-muted-foreground"}`}
              >
                {severity}: {count}
              </div>
            ))}
          {!stats && (
            <span className="text-muted-foreground text-xs">
              Stats unavailable — is the backend running?
            </span>
          )}
        </div>
      </section>

      {!data || data.items.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
          No problem mentions yet. Extract problems from evidence to populate this table.
        </div>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border bg-card/70">
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
                      &ldquo;{item.quote_text}&rdquo;
                    </p>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-block rounded-full border px-2 py-0.5 text-xs font-medium ${SEVERITY_COLORS[item.severity] ?? ""}`}
                    >
                      {item.severity}
                    </span>
                  </td>
                  <td className="px-4 py-3">{item.persona ?? "—"}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {item.tags.length ? item.tags.join(", ") : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="border-t border-border bg-muted/30 px-4 py-2 text-xs text-muted-foreground">
            Showing {data.items.length} of {data.total} problems · Page {data.page} of{" "}
            {data.total_pages}
          </div>
        </div>
      )}
    </div>
  );
}

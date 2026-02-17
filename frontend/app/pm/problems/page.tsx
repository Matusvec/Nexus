import PageHeader from "@/components/pm/PageHeader";
import { SeverityBadge } from "@/components/pm/shared/SeverityBadge";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type {
  ProblemMention,
  ProblemStats,
  PaginatedResponse,
} from "@/lib/pm/types";
import { AlertTriangle } from "lucide-react";

const SEVERITY_ORDER = ["critical", "high", "medium", "low"] as const;

export default async function ProblemsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = await searchParams;
  const page = Number(sp.page ?? 1);
  const severity = typeof sp.severity === "string" ? sp.severity : undefined;
  const persona = typeof sp.persona === "string" ? sp.persona : undefined;
  const search = typeof sp.search === "string" ? sp.search : undefined;
  const perPage = 20;

  const qs = new URLSearchParams();
  qs.set("page", String(page));
  qs.set("per_page", String(perPage));
  if (severity) qs.set("severity", severity);
  if (persona) qs.set("persona", persona);
  if (search) qs.set("search", search);

  const [data, stats] = await Promise.all([
    pmFetchSafe<PaginatedResponse<ProblemMention>>(
      `/problems?${qs.toString()}`
    ),
    pmFetchSafe<ProblemStats>("/problems/stats"),
  ]);

  const hasFilters = !!(severity || persona || search);

  function buildFilterUrl(
    overrides: Record<string, string | undefined>
  ): string {
    const p = new URLSearchParams();
    const s = overrides.severity ?? severity;
    const pe = overrides.persona ?? persona;
    const se = overrides.search ?? search;
    if (s) p.set("severity", s);
    if (pe) p.set("persona", pe);
    if (se) p.set("search", se);
    return `/pm/problems?${p.toString()}`;
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Problems"
        description="Extracted problem mentions with severity and traceability."
      />

      {/* Severity distribution bar */}
      {stats && (
        <div className="rounded-2xl border border-border bg-card p-4">
          <p className="text-[11px] font-medium uppercase tracking-[0.1em] text-muted-foreground mb-3">
            Severity Distribution
          </p>
          <div className="flex flex-wrap gap-2">
            {SEVERITY_ORDER.map((sev) => {
              const count = stats.by_severity?.[sev] ?? 0;
              if (count === 0) return null;
              const isActive = severity === sev;
              return (
                <Link
                  key={sev}
                  href={buildFilterUrl({
                    severity: isActive ? undefined : sev,
                  })}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors duration-150 ${
                    isActive
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border hover:bg-muted"
                  }`}
                >
                  <SeverityBadge severity={sev as ProblemMention["severity"]} />
                  <span className="tabular-nums">{count}</span>
                </Link>
              );
            })}
            {stats.total > 0 && (
              <span className="flex items-center text-xs text-muted-foreground ml-2">
                {stats.total} total
              </span>
            )}
          </div>
        </div>
      )}

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Persona filter from stats */}
        {stats &&
          stats.by_persona &&
          Object.keys(stats.by_persona).length > 0 && (
            <div className="flex gap-1">
              {Object.entries(stats.by_persona)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 6)
                .map(([p, count]) => {
                  const isActive = persona === p;
                  return (
                    <Link
                      key={p}
                      href={buildFilterUrl({
                        persona: isActive ? undefined : p,
                      })}
                      className={`rounded-lg border px-2.5 py-1 text-xs transition-colors duration-150 ${
                        isActive
                          ? "border-primary bg-primary/10 text-primary font-medium"
                          : "border-border text-muted-foreground hover:bg-muted"
                      }`}
                    >
                      {p} ({count})
                    </Link>
                  );
                })}
            </div>
          )}
        {hasFilters && (
          <Link
            href="/pm/problems"
            className="text-xs text-primary hover:underline"
          >
            Clear filters
          </Link>
        )}
      </div>

      {!data || data.items.length === 0 ? (
        <EmptyState
          icon={AlertTriangle}
          title={hasFilters ? "No matching problems" : "No problems yet"}
          description={
            hasFilters
              ? "Try adjusting your filters or clearing them."
              : "Extract problems from evidence to populate this table."
          }
          actionLabel={hasFilters ? "Clear Filters" : "View Evidence"}
          actionHref={hasFilters ? "/pm/problems" : "/pm/evidence"}
        />
      ) : (
        <>
          <div className="overflow-hidden rounded-2xl border border-border bg-card">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-left text-[11px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
                <tr>
                  <th className="px-4 py-3">Problem</th>
                  <th className="px-4 py-3 w-28">Severity</th>
                  <th className="px-4 py-3">Persona</th>
                  <th className="px-4 py-3">Tags</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item) => (
                  <tr
                    key={item.id}
                    className="border-t border-border transition-colors duration-100 hover:bg-muted/30"
                  >
                    <td className="px-4 py-3">
                      <Link
                        href={`/pm/problems/${item.id}`}
                        className="font-medium text-foreground hover:text-primary hover:underline"
                      >
                        {item.problem_statement}
                      </Link>
                      <p className="mt-1 text-xs text-muted-foreground line-clamp-1">
                        &ldquo;{item.quote_text}&rdquo;
                      </p>
                    </td>
                    <td className="px-4 py-3">
                      <SeverityBadge severity={item.severity} />
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {item.persona ?? "—"}
                    </td>
                    <td className="px-4 py-3">
                      {item.tags.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {item.tags.map((tag) => (
                            <span
                              key={tag}
                              className="rounded-lg bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {data.total_pages > 1 && (
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <span>
                Showing {(page - 1) * perPage + 1}–
                {Math.min(page * perPage, data.total)} of {data.total}
              </span>
              <div className="flex gap-2">
                {page > 1 && (
                  <Link
                    href={`/pm/problems?page=${page - 1}${severity ? `&severity=${severity}` : ""}${persona ? `&persona=${persona}` : ""}`}
                    className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
                  >
                    ← Previous
                  </Link>
                )}
                {page < data.total_pages && (
                  <Link
                    href={`/pm/problems?page=${page + 1}${severity ? `&severity=${severity}` : ""}${persona ? `&persona=${persona}` : ""}`}
                    className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
                  >
                    Next →
                  </Link>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

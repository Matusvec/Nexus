import Link from "next/link";
import PageHeader from "@/components/pm/PageHeader";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import { pmFetchSafe } from "@/lib/pm/api";
import type { Evidence, PaginatedResponse } from "@/lib/pm/types";
import { FileText, Upload } from "lucide-react";

export default async function EvidencePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = await searchParams;
  const page = Number(sp.page ?? 1);
  const perPage = 20;

  const data = await pmFetchSafe<PaginatedResponse<Evidence>>(
    `/evidence?page=${page}&per_page=${perPage}`,
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Evidence"
        description="Upload and inspect raw customer signal. Each item is chunked and traceable."
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

      {!data || data.items.length === 0 ? (
        <EmptyState
          icon={FileText}
          title="No evidence yet"
          description="Upload your first transcript, support log, or survey response to start the pipeline."
          actionLabel="Upload Evidence"
          actionHref="/pm/evidence/upload"
        />
      ) : (
        <>
          <div className="overflow-hidden rounded-2xl border border-border bg-card">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-left text-[11px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
                <tr>
                  <th className="px-4 py-3">Title</th>
                  <th className="px-4 py-3">Source</th>
                  <th className="px-4 py-3">Persona</th>
                  <th className="px-4 py-3">Segment</th>
                  <th className="px-4 py-3 text-right">Chunks</th>
                  <th className="px-4 py-3">Created</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item) => (
                  <tr
                    key={item.id}
                    className="border-t border-border transition-colors duration-100 hover:bg-muted/30"
                  >
                    <td className="px-4 py-3 font-medium">
                      <Link
                        href={`/pm/evidence/${item.id}`}
                        className="text-foreground hover:text-primary hover:underline"
                      >
                        {item.title}
                      </Link>
                    </td>
                    <td className="px-4 py-3">
                      <span className="rounded-lg bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        {item.source_type.replace("_", " ")}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {item.persona ?? "—"}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {item.segment ?? "—"}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">
                      {item.chunk_count}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground tabular-nums">
                      {item.created_at
                        ? new Date(item.created_at).toLocaleDateString()
                        : "—"}
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
                    href={`/pm/evidence?page=${page - 1}`}
                    className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
                  >
                    ← Previous
                  </Link>
                )}
                {page < data.total_pages && (
                  <Link
                    href={`/pm/evidence?page=${page + 1}`}
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

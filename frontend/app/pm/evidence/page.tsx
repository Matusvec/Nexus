import Link from "next/link";
import PageHeader from "@/components/pm/PageHeader";
import { pmFetchSafe } from "@/lib/pm/api";
import type { Evidence, PaginatedResponse } from "@/lib/pm/types";

export default async function EvidencePage() {
  const data = await pmFetchSafe<PaginatedResponse<Evidence>>(
    "/evidence?page=1&per_page=20",
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Evidence"
        description="Upload and inspect raw customer signal. Each item is chunked and traceable."
        actions={
          <Link
            href="/pm/evidence/upload"
            className="rounded-full bg-[hsl(var(--primary))] px-4 py-2 text-sm font-semibold text-primary-foreground"
          >
            Upload Evidence
          </Link>
        }
      />

      {!data || data.items.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
          No evidence yet. Upload your first transcript to start the pipeline.
        </div>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border bg-card/70">
          <table className="w-full text-sm">
            <thead className="bg-muted/70 text-left text-xs uppercase tracking-[0.2em] text-muted-foreground">
              <tr>
                <th className="px-4 py-3">Title</th>
                <th className="px-4 py-3">Source</th>
                <th className="px-4 py-3">Persona</th>
                <th className="px-4 py-3">Segment</th>
                <th className="px-4 py-3">Chunks</th>
                <th className="px-4 py-3">Created</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((item) => (
                <tr key={item.id} className="border-t border-border">
                  <td className="px-4 py-3 font-medium">
                    <Link href={`/pm/evidence/${item.id}`} className="hover:underline">
                      {item.title}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {item.source_type}
                  </td>
                  <td className="px-4 py-3">{item.persona ?? "—"}</td>
                  <td className="px-4 py-3">{item.segment ?? "—"}</td>
                  <td className="px-4 py-3">{item.chunk_count}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {item.created_at
                      ? new Date(item.created_at).toLocaleDateString()
                      : "—"}
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

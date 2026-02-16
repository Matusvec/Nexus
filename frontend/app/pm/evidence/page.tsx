import Link from "next/link";
import PageHeader from "@/components/pm/PageHeader";

interface EvidenceItem {
  id: string;
  title: string;
  source_type: string;
  persona: string | null;
  segment: string | null;
  chunk_count: number;
  created_at: string | null;
}

interface EvidenceListResponse {
  items: EvidenceItem[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

async function loadEvidence(): Promise<EvidenceListResponse | null> {
  try {
    const res = await fetch("/api/v1/evidence?page=1&per_page=20", {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as EvidenceListResponse;
  } catch {
    return null;
  }
}

export default async function EvidencePage() {
  const data = await loadEvidence();

  return (
    <div className="space-y-6">
      <PageHeader
        title="Evidence"
        description="Upload and inspect raw customer signal. Each item is chunked and traceable."
        actions={
          <Link
            href="/pm/evidence/upload"
            className="rounded-full bg-[hsl(var(--primary))] px-4 py-2 text-sm font-semibold text-white"
          >
            Upload Evidence
          </Link>
        }
      />

      {!data || data.items.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
          No evidence yet. Upload your first transcript to start the pipeline.
        </div>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-border bg-white/70">
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
                  <td className="px-4 py-3">{item.persona ?? "--"}</td>
                  <td className="px-4 py-3">{item.segment ?? "--"}</td>
                  <td className="px-4 py-3">{item.chunk_count}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {item.created_at ? new Date(item.created_at).toLocaleDateString() : "--"}
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

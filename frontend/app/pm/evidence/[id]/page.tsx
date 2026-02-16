import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";

interface EvidenceChunk {
  id: string;
  chunk_index: number;
  chunk_text: string;
  start_offset: number;
  end_offset: number;
  token_count: number | null;
}

interface EvidenceDetail {
  id: string;
  title: string;
  source_type: string;
  persona: string | null;
  segment: string | null;
  source_date: string | null;
  chunk_count: number;
  created_at: string | null;
  raw_text: string;
  chunks: EvidenceChunk[];
}

async function loadEvidence(id: string): Promise<EvidenceDetail | null> {
  try {
    const res = await fetch(`/api/v1/evidence/${id}`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as EvidenceDetail;
  } catch {
    return null;
  }
}

export default async function EvidenceDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const evidence = await loadEvidence(params.id);

  if (!evidence) {
    return (
      <div className="space-y-6">
        <PageHeader title="Evidence Detail" />
        <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
          Evidence not found or backend unavailable.
        </div>
        <Link href="/pm/evidence" className="text-sm text-primary underline">
          Back to Evidence
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title={evidence.title}
        description={`${evidence.source_type} • ${evidence.persona ?? "Unknown persona"} • ${evidence.segment ?? "Unknown segment"}`}
      />

      <section className="rounded-2xl border border-border bg-white/70 p-6">
        <h2 className="text-lg font-semibold">Raw Text</h2>
        <p className="mt-3 whitespace-pre-wrap text-sm text-muted-foreground">
          {evidence.raw_text}
        </p>
      </section>

      <section className="rounded-2xl border border-border bg-white/70 p-6">
        <h2 className="text-lg font-semibold">Chunks</h2>
        <div className="mt-4 space-y-4">
          {evidence.chunks.map((chunk) => (
            <div
              key={chunk.id}
              className="rounded-xl border border-border bg-white p-4"
            >
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Chunk {chunk.chunk_index + 1}</span>
                <span>
                  {chunk.token_count ? `${chunk.token_count} tokens` : "--"}
                </span>
              </div>
              <p className="mt-2 text-sm text-foreground/80">
                {chunk.chunk_text}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

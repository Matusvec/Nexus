import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";
import { pmFetchSafe } from "@/lib/pm/api";
import type { EvidenceDetail } from "@/lib/pm/types";

export default async function EvidenceDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const evidence = await pmFetchSafe<EvidenceDetail>(`/evidence/${id}`);

  if (!evidence) {
    return (
      <div className="space-y-6">
        <PageHeader title="Evidence Detail" />
        <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
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
        description={`${evidence.source_type} · ${evidence.persona ?? "Unknown persona"} · ${evidence.segment ?? "Unknown segment"}`}
      />

      <section className="rounded-2xl border border-border bg-card/70 p-6">
        <h2 className="text-lg font-semibold">Raw Text</h2>
        <p className="mt-3 whitespace-pre-wrap text-sm text-muted-foreground">
          {evidence.raw_text}
        </p>
      </section>

      <section className="rounded-2xl border border-border bg-card/70 p-6">
        <h2 className="text-lg font-semibold">Chunks ({evidence.chunks.length})</h2>
        <div className="mt-4 space-y-4">
          {evidence.chunks.map((chunk) => (
            <div
              key={chunk.id}
              className="rounded-xl border border-border bg-card p-4"
            >
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Chunk {chunk.chunk_index + 1}</span>
                <span>
                  {chunk.token_count ? `${chunk.token_count} tokens` : "—"}
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

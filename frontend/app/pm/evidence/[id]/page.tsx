import PageHeader from "@/components/pm/PageHeader";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import { SeverityBadge } from "@/components/pm/shared/SeverityBadge";
import { pmFetchSafe } from "@/lib/pm/api";
import type { EvidenceDetail, ProblemMention, PaginatedResponse } from "@/lib/pm/types";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { FileQuestion, AlertTriangle } from "lucide-react";
import Link from "next/link";

export default async function EvidenceDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const [evidence, problems] = await Promise.all([
    pmFetchSafe<EvidenceDetail>(`/evidence/${id}`),
    pmFetchSafe<PaginatedResponse<ProblemMention>>(
      `/problems?evidence_id=${id}&page=1&per_page=100`
    ),
  ]);

  if (!evidence) {
    return (
      <div className="space-y-6">
        <PageHeader
          title="Evidence Detail"
          backLabel="Back to Evidence"
          backHref="/pm/evidence"
        />
        <EmptyState
          icon={FileQuestion}
          title="Evidence not found"
          description="This evidence may have been deleted, or the backend is unavailable."
          actionLabel="Back to Evidence"
          actionHref="/pm/evidence"
        />
      </div>
    );
  }

  const extractedProblems = problems?.items ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        title={evidence.title}
        description={`${evidence.source_type.replace("_", " ")} · ${evidence.persona ?? "Unknown persona"} · ${evidence.segment ?? "Unknown segment"}`}
        backLabel="Back to Evidence"
        backHref="/pm/evidence"
      />

      {/* Metadata bar */}
      <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
        <span>
          <span className="text-[11px] font-medium uppercase tracking-[0.1em]">
            Chunks
          </span>{" "}
          <span className="font-semibold text-foreground">
            {evidence.chunks.length}
          </span>
        </span>
        <span>
          <span className="text-[11px] font-medium uppercase tracking-[0.1em]">
            Problems
          </span>{" "}
          <span className="font-semibold text-foreground">
            {extractedProblems.length}
          </span>
        </span>
        {evidence.source_date && (
          <span>
            <span className="text-[11px] font-medium uppercase tracking-[0.1em]">
              Date
            </span>{" "}
            <span className="font-semibold text-foreground">
              {new Date(evidence.source_date).toLocaleDateString()}
            </span>
          </span>
        )}
        {evidence.created_at && (
          <span>
            <span className="text-[11px] font-medium uppercase tracking-[0.1em]">
              Uploaded
            </span>{" "}
            <span className="font-semibold text-foreground">
              {new Date(evidence.created_at).toLocaleDateString()}
            </span>
          </span>
        )}
      </div>

      <Tabs defaultValue="raw" className="space-y-4">
        <TabsList>
          <TabsTrigger value="raw">Raw Text</TabsTrigger>
          <TabsTrigger value="problems">
            Problems ({extractedProblems.length})
          </TabsTrigger>
          <TabsTrigger value="chunks">
            Chunks ({evidence.chunks.length})
          </TabsTrigger>
        </TabsList>

        {/* Raw Text Tab */}
        <TabsContent value="raw">
          <div className="rounded-2xl border border-border bg-card p-6">
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground/80">
              {evidence.raw_text}
            </p>
          </div>
        </TabsContent>

        {/* Extracted Problems Tab */}
        <TabsContent value="problems">
          {extractedProblems.length === 0 ? (
            <EmptyState
              icon={AlertTriangle}
              title="No problems extracted"
              description="Run extraction from the evidence list to find problems in this document."
            />
          ) : (
            <div className="space-y-3">
              {extractedProblems.map((p) => (
                <Link
                  key={p.id}
                  href={`/pm/problems/${p.id}`}
                  className="block rounded-2xl border border-border bg-card p-4 transition-colors duration-150 hover:bg-muted/30"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium">
                        {p.problem_statement}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                        &ldquo;{p.quote_text}&rdquo;
                      </p>
                    </div>
                    <SeverityBadge severity={p.severity} />
                  </div>
                  {p.tags.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {p.tags.map((tag) => (
                        <span
                          key={tag}
                          className="rounded-lg bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </Link>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Chunks Tab */}
        <TabsContent value="chunks">
          <div className="space-y-3">
            {evidence.chunks.map((chunk) => (
              <div
                key={chunk.id}
                className="rounded-2xl border border-border bg-card p-4"
              >
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span className="font-medium">
                    Chunk {chunk.chunk_index + 1}
                  </span>
                  <span className="tabular-nums">
                    {chunk.token_count
                      ? `${chunk.token_count} tokens`
                      : `chars ${chunk.start_offset}–${chunk.end_offset}`}
                  </span>
                </div>
                <p className="mt-2 text-sm text-foreground/80 leading-relaxed">
                  {chunk.chunk_text}
                </p>
              </div>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

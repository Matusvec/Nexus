// ============================================
// SimilarProblemsPanel — Strategy §4.5
// ============================================
// Sheet (slide-over) showing similar problems with similarity scores.

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { getSimilarProblems } from "@/lib/pm/api";
import { SeverityBadge, type Severity } from "@/components/pm/shared/SeverityBadge";
import { Loader2 } from "lucide-react";
import type { SimilarProblem } from "@/lib/pm/types";

interface SimilarProblemsPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  queryText: string;
}

export function SimilarProblemsPanel({
  open,
  onOpenChange,
  queryText,
}: SimilarProblemsPanelProps) {
  const [results, setResults] = useState<SimilarProblem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || !queryText) return;
    setLoading(true);
    getSimilarProblems(queryText, 10)
      .then((data) => setResults(data.results))
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, [open, queryText]);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Similar Problems</SheetTitle>
          <SheetDescription className="line-clamp-2">
            Problems similar to: &ldquo;{queryText}&rdquo;
          </SheetDescription>
        </SheetHeader>
        <div className="mt-6 space-y-3">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-primary" strokeWidth={1.75} />
            </div>
          )}
          {!loading && results.length === 0 && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No similar problems found.
            </p>
          )}
          {!loading &&
            results.map((result) => (
              <Link
                key={result.problem.id}
                href={`/pm/problems/${result.problem.id}`}
                onClick={() => onOpenChange(false)}
                className="block rounded-xl border border-border p-3 transition-colors duration-150 hover:bg-muted/50"
              >
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm font-medium text-foreground leading-tight">
                    {result.problem.problem_statement}
                  </p>
                  <span className="shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary">
                    {Math.round(result.score * 100)}%
                  </span>
                </div>
                {result.problem.severity && (
                  <div className="mt-2">
                    <SeverityBadge
                      severity={result.problem.severity.toLowerCase() as Severity}
                    />
                  </div>
                )}
              </Link>
            ))}
        </div>
      </SheetContent>
    </Sheet>
  );
}

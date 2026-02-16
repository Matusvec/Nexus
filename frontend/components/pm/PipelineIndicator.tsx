"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export type PipelineStatus = "complete" | "running" | "pending";

export interface PipelineStep {
  label: string;
  count?: number | null;
  status: PipelineStatus;
}

const statusClasses: Record<PipelineStatus, string> = {
  complete: "bg-emerald-500",
  running: "bg-amber-500",
  pending: "bg-slate-300 dark:bg-slate-600",
};

function deriveStatus(count: number | null): PipelineStatus {
  if (count === null || count === undefined) return "pending";
  return count > 0 ? "complete" : "pending";
}

export default function PipelineIndicator() {
  const [steps, setSteps] = useState<PipelineStep[]>([
    { label: "Evidence", count: null, status: "pending" },
    { label: "Problems", count: null, status: "pending" },
    { label: "Clusters", count: null, status: "pending" },
    { label: "Proposals", count: null, status: "pending" },
    { label: "Tasks", count: null, status: "pending" },
    { label: "Roadmap", count: null, status: "pending" },
  ]);

  useEffect(() => {
    async function fetchCounts() {
      try {
        const [evRes, prRes, clRes, rmRes] = await Promise.allSettled([
          fetch("/api/v1/evidence?page=1&per_page=1").then((r) =>
            r.ok ? r.json() : null,
          ),
          fetch("/api/v1/problems?page=1&per_page=1").then((r) =>
            r.ok ? r.json() : null,
          ),
          fetch("/api/v1/clusters?page=1&per_page=1").then((r) =>
            r.ok ? r.json() : null,
          ),
          fetch("/api/v1/roadmap").then((r) => (r.ok ? r.json() : null)),
        ]);

        const ev =
          evRes.status === "fulfilled" ? evRes.value?.total ?? null : null;
        const pr =
          prRes.status === "fulfilled" ? prRes.value?.total ?? null : null;
        const cl =
          clRes.status === "fulfilled" ? clRes.value?.total ?? null : null;
        const rm =
          rmRes.status === "fulfilled"
            ? rmRes.value?.items?.length ?? null
            : null;

        // Proposals = roadmap items (each roadmap item wraps a proposal)
        const proposals = rm;

        setSteps([
          { label: "Evidence", count: ev, status: deriveStatus(ev) },
          { label: "Problems", count: pr, status: deriveStatus(pr) },
          { label: "Clusters", count: cl, status: deriveStatus(cl) },
          {
            label: "Proposals",
            count: proposals,
            status: deriveStatus(proposals),
          },
          { label: "Tasks", count: null, status: "pending" },
          { label: "Roadmap", count: rm, status: deriveStatus(rm) },
        ]);
      } catch {
        // keep defaults on error
      }
    }
    fetchCounts();
  }, []);

  return (
    <div className="rounded-2xl border border-border bg-card/80 px-4 py-3 shadow-sm">
      <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.2em] text-muted-foreground">
        Pipeline
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        {steps.map((step, index) => (
          <motion.div
            key={step.label}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="flex items-center gap-2"
          >
            <div
              className={cn(
                "h-2 w-2 rounded-full",
                statusClasses[step.status],
              )}
            />
            <div className="text-sm font-medium text-foreground">
              {step.label}
            </div>
            {step.count !== undefined && step.count !== null && (
              <div className="rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                {step.count}
              </div>
            )}
            {index < steps.length - 1 && (
              <div className="h-px w-6 bg-border" />
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}

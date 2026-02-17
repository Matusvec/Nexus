"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export type PipelineStatus = "complete" | "running" | "pending";

export interface PipelineStep {
  label: string;
  count?: number | null;
  status: PipelineStatus;
  href: string;
}

const statusClasses: Record<PipelineStatus, string> = {
  complete: "bg-green-500",
  running: "bg-amber-500 animate-pulse",
  pending: "bg-slate-300",
};

function deriveStatus(count: number | null): PipelineStatus {
  if (count === null || count === undefined) return "pending";
  return count > 0 ? "complete" : "pending";
}

export default function PipelineIndicator() {
  const router = useRouter();
  const pathname = usePathname();

  const [steps, setSteps] = useState<PipelineStep[]>([
    { label: "Evidence", count: null, status: "pending", href: "/pm/evidence" },
    { label: "Problems", count: null, status: "pending", href: "/pm/problems" },
    { label: "Clusters", count: null, status: "pending", href: "/pm/clusters" },
    { label: "Proposals", count: null, status: "pending", href: "/pm/proposals" },
    { label: "Tasks", count: null, status: "pending", href: "/pm/tasks" },
    { label: "Roadmap", count: null, status: "pending", href: "/pm/roadmap" },
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

        const proposals = rm;

        setSteps([
          { label: "Evidence", count: ev, status: deriveStatus(ev), href: "/pm/evidence" },
          { label: "Problems", count: pr, status: deriveStatus(pr), href: "/pm/problems" },
          { label: "Clusters", count: cl, status: deriveStatus(cl), href: "/pm/clusters" },
          { label: "Proposals", count: proposals, status: deriveStatus(proposals), href: "/pm/proposals" },
          { label: "Tasks", count: null, status: "pending", href: "/pm/tasks" },
          { label: "Roadmap", count: rm, status: deriveStatus(rm), href: "/pm/roadmap" },
        ]);
      } catch {
        // keep defaults on error
      }
    }
    fetchCounts();
  }, []);

  return (
    <div
      className="rounded-2xl border border-border bg-card/80 px-4 py-3 shadow-sm"
      aria-live="polite"
      role="navigation"
      aria-label="Pipeline progress"
    >
      <div className="flex flex-wrap items-center gap-2 text-[11px] font-medium uppercase tracking-[0.15em] text-muted-foreground">
        Pipeline
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        {steps.map((step, index) => {
          const isActive = pathname.startsWith(step.href);
          return (
            <motion.div
              key={step.label}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="flex items-center gap-2"
            >
              <button
                onClick={() => router.push(step.href)}
                title={`Click to view ${step.label}`}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-2 py-1 transition-colors duration-150 hover:bg-muted/50",
                  isActive && "bg-muted/80"
                )}
              >
                <div
                  className={cn(
                    "h-2 w-2 rounded-full",
                    statusClasses[step.status],
                  )}
                />
                <span
                  className={cn(
                    "text-sm font-medium",
                    isActive ? "text-foreground" : "text-muted-foreground"
                  )}
                >
                  {step.label}
                </span>
                {step.count !== undefined && step.count !== null && (
                  <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                    {step.count}
                  </span>
                )}
              </button>
              {index < steps.length - 1 && (
                <div className="h-px w-6 bg-border" />
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

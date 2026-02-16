"use client";

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
  pending: "bg-slate-300",
};

const stepsDefault: PipelineStep[] = [
  { label: "Evidence", count: 0, status: "complete" },
  { label: "Problems", count: 0, status: "complete" },
  { label: "Clusters", count: null, status: "running" },
  { label: "Proposals", count: null, status: "pending" },
  { label: "Tasks", count: null, status: "pending" },
  { label: "Roadmap", count: null, status: "pending" },
];

export default function PipelineIndicator({
  steps = stepsDefault,
}: {
  steps?: PipelineStep[];
}) {
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
                statusClasses[step.status]
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

// ============================================
// JobProgress — Strategy §5
// ============================================
// Inline progress bar with polling. Displays job status and auto-updates.

"use client";

import { useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import { getJobStatus, pmKeys } from "@/lib/pm/api";
import { useJobsStore } from "@/lib/pm/store";
import { cn } from "@/lib/utils";

interface JobProgressProps {
  jobId: string;
  label: string;
  onComplete?: () => void;
  onError?: (error: string) => void;
  className?: string;
}

export function JobProgress({
  jobId,
  label,
  onComplete,
  onError,
  className,
}: JobProgressProps) {
  const { setJob, removeJob } = useJobsStore();

  const { data: job } = useQuery({
    queryKey: pmKeys.jobs.detail(jobId),
    queryFn: () => getJobStatus(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed") return false;
      return 2000; // Poll every 2 seconds
    },
  });

  const handleComplete = useCallback(() => {
    removeJob(jobId);
    toast.success(`${label} completed successfully.`);
    onComplete?.();
  }, [jobId, label, onComplete, removeJob]);

  const handleError = useCallback(
    (error: string) => {
      removeJob(jobId);
      toast.error(`${label} failed: ${error}`);
      onError?.(error);
    },
    [jobId, label, onError, removeJob]
  );

  useEffect(() => {
    if (!job) return;

    setJob(jobId, job);

    if (job.status === "completed") {
      handleComplete();
    } else if (job.status === "failed") {
      handleError(job.error ?? "Unknown error");
    }
  }, [job, jobId, setJob, handleComplete, handleError]);

  const isRunning = !job || job.status === "pending" || job.status === "running";
  const isComplete = job?.status === "completed";
  const isFailed = job?.status === "failed";

  return (
    <div className={cn("flex items-center gap-3", className)}>
      {isRunning && (
        <>
          <Loader2 className="h-4 w-4 animate-spin text-primary" strokeWidth={1.75} />
          <span className="text-sm text-muted-foreground">{label}</span>
          <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
            <div className="h-full bg-primary rounded-full animate-pulse w-2/3 transition-all duration-300" />
          </div>
        </>
      )}
      {isComplete && (
        <>
          <CheckCircle2 className="h-4 w-4 text-green-600" strokeWidth={1.75} />
          <span className="text-sm text-green-700">{label} — Complete</span>
        </>
      )}
      {isFailed && (
        <>
          <XCircle className="h-4 w-4 text-red-600" strokeWidth={1.75} />
          <span className="text-sm text-red-700">{label} — Failed</span>
        </>
      )}
    </div>
  );
}

"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Sparkles, Layers, Loader2 } from "lucide-react";
import { embedProblems, runClustering, getJobStatus } from "@/lib/pm/api";

type Step = "idle" | "embedding" | "clustering" | "done" | "error";

export default function RunPipelineButton() {
  const router = useRouter();
  const [step, setStep] = useState<Step>("idle");
  const [message, setMessage] = useState("");

  async function pollJob(jobId: string, maxWait = 120_000): Promise<boolean> {
    const start = Date.now();
    while (Date.now() - start < maxWait) {
      await new Promise((r) => setTimeout(r, 2000));
      try {
        const status = await getJobStatus(jobId);
        if (status.status === "completed") return true;
        if (status.status === "failed") {
          setMessage(status.error ?? "Job failed");
          return false;
        }
      } catch {
        // retry
      }
    }
    setMessage("Job timed out");
    return false;
  }

  async function handleRun() {
    setStep("embedding");
    setMessage("Embedding un-embedded problems…");

    try {
      const embedJob = await embedProblems();
      const embedOk = await pollJob(embedJob.job_id);
      if (!embedOk) {
        setStep("error");
        return;
      }

      setStep("clustering");
      setMessage("Running clustering…");
      const result = await runClustering(0.75);
      setMessage(`Created ${result.clusters_created} cluster(s)`);
      setStep("done");
      router.refresh();
    } catch (err) {
      console.error(err);
      setMessage("Pipeline failed. Check the console for details.");
      setStep("error");
    }
  }

  const isRunning = step === "embedding" || step === "clustering";

  return (
    <div className="flex items-center gap-3">
      {message && (
        <span
          className={`text-sm ${
            step === "error"
              ? "text-destructive"
              : step === "done"
                ? "text-green-600 dark:text-green-400"
                : "text-muted-foreground"
          }`}
        >
          {message}
        </span>
      )}
      <button
        onClick={handleRun}
        disabled={isRunning}
        className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/90 active:scale-[0.98] disabled:opacity-50"
      >
        {isRunning ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : step === "done" ? (
          <Layers className="h-4 w-4" strokeWidth={1.75} />
        ) : (
          <Sparkles className="h-4 w-4" strokeWidth={1.75} />
        )}
        {step === "embedding"
          ? "Embedding…"
          : step === "clustering"
            ? "Clustering…"
            : step === "done"
              ? "Re-run Pipeline"
              : "Embed & Cluster"}
      </button>
    </div>
  );
}

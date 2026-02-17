// ============================================
// ProposalActions — Strategy §4.10
// ============================================
// Approve / Reject / Regenerate / Generate Tasks button group.

"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Check, X, RefreshCw, Hammer } from "lucide-react";
import { toast } from "sonner";
import {
  approveProposal,
  rejectProposal,
  regenerateProposal,
  generateTasks,
} from "@/lib/pm/api";
import { ConfirmDialog } from "@/components/pm/shared/ConfirmDialog";
import type { ProposalStatus } from "@/lib/pm/types";
import { cn } from "@/lib/utils";

interface ProposalActionsProps {
  proposalId: string;
  status: ProposalStatus;
  onStatusChange?: () => void;
  tasksGenerated?: boolean;
}

export function ProposalActions({
  proposalId,
  status,
  onStatusChange,
  tasksGenerated,
}: ProposalActionsProps) {
  const router = useRouter();
  const [dialog, setDialog] = useState<
    "approve" | "reject" | "regenerate" | null
  >(null);
  const [loading, setLoading] = useState(false);

  const handleApprove = async () => {
    setLoading(true);
    try {
      await approveProposal(proposalId);
      toast.success("Proposal approved.");
      onStatusChange?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Approval failed");
    } finally {
      setLoading(false);
    }
  };

  const handleReject = async () => {
    setLoading(true);
    try {
      await rejectProposal(proposalId);
      toast.success("Proposal rejected.");
      onStatusChange?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Rejection failed");
    } finally {
      setLoading(false);
    }
  };

  const handleRegenerate = async () => {
    setLoading(true);
    try {
      await regenerateProposal(proposalId);
      toast.success("Regeneration started. This may take a moment.");
      onStatusChange?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Regeneration failed");
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateTasks = async () => {
    try {
      const job = await generateTasks(proposalId);
      toast.success("Task generation started.");
      if (job.job_id) {
        router.push(`/pm/proposals/${proposalId}`);
      }
      onStatusChange?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Task generation failed");
    }
  };

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        {status !== "approved" && (
          <button
            onClick={() => setDialog("approve")}
            className="inline-flex items-center gap-1.5 rounded-xl bg-green-600 px-3 py-2 text-sm font-medium text-white transition-colors duration-150 hover:bg-green-700 active:scale-[0.98]"
          >
            <Check className="h-4 w-4" strokeWidth={1.75} />
            Approve
          </button>
        )}

        {status !== "rejected" && (
          <button
            onClick={() => setDialog("reject")}
            className="inline-flex items-center gap-1.5 rounded-xl border border-border px-3 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted active:scale-[0.98]"
          >
            <X className="h-4 w-4" strokeWidth={1.75} />
            Reject
          </button>
        )}

        {status === "draft" && (
          <button
            onClick={() => setDialog("regenerate")}
            className="inline-flex items-center gap-1.5 rounded-xl border border-border px-3 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted active:scale-[0.98]"
          >
            <RefreshCw className="h-4 w-4" strokeWidth={1.75} />
            Regenerate
          </button>
        )}

        {status === "approved" && !tasksGenerated && (
          <button
            onClick={handleGenerateTasks}
            className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/90 active:scale-[0.98]"
          >
            <Hammer className="h-4 w-4" strokeWidth={1.75} />
            Generate Tasks
          </button>
        )}
      </div>

      <ConfirmDialog
        open={dialog === "approve"}
        onOpenChange={(open) => !open && setDialog(null)}
        title="Approve Proposal"
        description="This proposal will appear in the roadmap and can have tasks generated."
        confirmLabel="Approve"
        onConfirm={handleApprove}
        loading={loading}
      />

      <ConfirmDialog
        open={dialog === "reject"}
        onOpenChange={(open) => !open && setDialog(null)}
        title="Reject Proposal"
        description="This proposal will be marked as rejected and grayed out."
        confirmLabel="Reject"
        variant="destructive"
        onConfirm={handleReject}
        loading={loading}
      />

      <ConfirmDialog
        open={dialog === "regenerate"}
        onOpenChange={(open) => !open && setDialog(null)}
        title="Regenerate Proposal"
        description="This will replace the current specification with a newly generated one. This action cannot be undone."
        confirmLabel="Regenerate"
        variant="destructive"
        onConfirm={handleRegenerate}
        loading={loading}
      />
    </>
  );
}

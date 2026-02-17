"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Trash2, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { deleteEvidence } from "@/lib/pm/api";

interface DeleteEvidenceButtonProps {
  evidenceId: string;
  evidenceTitle: string;
  /** "icon" shows just an icon (for table rows), "button" shows full text */
  variant?: "icon" | "button";
}

export default function DeleteEvidenceButton({
  evidenceId,
  evidenceTitle,
  variant = "button",
}: DeleteEvidenceButtonProps) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  async function handleDelete() {
    setDeleting(true);
    try {
      await deleteEvidence(evidenceId);
      setOpen(false);
      router.push("/pm/evidence");
      router.refresh();
    } catch (err) {
      console.error("Failed to delete evidence:", err);
      setDeleting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {variant === "icon" ? (
          <button
            className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
            title="Delete evidence"
          >
            <Trash2 className="h-4 w-4" strokeWidth={1.75} />
          </button>
        ) : (
          <button className="inline-flex items-center gap-1.5 rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-2 text-sm font-medium text-destructive transition-colors duration-150 hover:bg-destructive/10 active:scale-[0.98]">
            <Trash2 className="h-4 w-4" strokeWidth={1.75} />
            Delete
          </button>
        )}
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete Evidence</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete &ldquo;{evidenceTitle}&rdquo;? This
            will also remove all associated chunks and extracted problems. This
            action cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <button
            onClick={() => setOpen(false)}
            disabled={deleting}
            className="rounded-xl border border-border px-4 py-2 text-sm font-medium transition-colors hover:bg-muted"
          >
            Cancel
          </button>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="inline-flex items-center gap-1.5 rounded-xl bg-destructive px-4 py-2 text-sm font-medium text-destructive-foreground transition-colors hover:bg-destructive/90 disabled:opacity-50"
          >
            {deleting ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4" strokeWidth={1.75} />
            )}
            {deleting ? "Deleting…" : "Delete"}
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

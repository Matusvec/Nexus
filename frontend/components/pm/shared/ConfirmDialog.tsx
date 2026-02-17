// ============================================
// ConfirmDialog — Strategy §5
// ============================================
// Confirmation modal for destructive or significant actions.

"use client";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface ConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "default" | "destructive";
  onConfirm: () => void;
  loading?: boolean;
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "default",
  onConfirm,
  loading = false,
}: ConfirmDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <DialogFooter className="flex gap-2 sm:justify-end">
          <button
            onClick={() => onOpenChange(false)}
            disabled={loading}
            className="rounded-xl border border-border px-4 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted"
          >
            {cancelLabel}
          </button>
          <button
            onClick={() => {
              onConfirm();
              if (!loading) onOpenChange(false);
            }}
            disabled={loading}
            className={cn(
              "rounded-xl px-4 py-2 text-sm font-medium transition-colors duration-150 active:scale-[0.98]",
              variant === "destructive"
                ? "bg-red-600 text-white hover:bg-red-700"
                : "bg-primary text-primary-foreground hover:bg-primary/90"
            )}
          >
            {loading ? "Processing…" : confirmLabel}
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

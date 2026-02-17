// ============================================
// StatusBadge — Strategy §5
// ============================================
// Renders a colored pill for proposal/job status.

import { cn } from "@/lib/utils";

export type Status = "draft" | "approved" | "rejected" | "archived" | "running" | "failed" | "completed";

const statusConfig: Record<
  Status,
  { label: string; className: string }
> = {
  draft: {
    label: "Draft",
    className: "bg-blue-100 text-blue-700 border-blue-200",
  },
  approved: {
    label: "Approved",
    className: "bg-green-100 text-green-700 border-green-200",
  },
  rejected: {
    label: "Rejected",
    className: "bg-gray-100 text-gray-500 border-gray-200",
  },
  archived: {
    label: "Archived",
    className: "bg-gray-100 text-gray-400 border-gray-200",
  },
  running: {
    label: "Running",
    className: "bg-amber-100 text-amber-700 border-amber-200 animate-pulse",
  },
  failed: {
    label: "Failed",
    className: "bg-red-100 text-red-700 border-red-200",
  },
  completed: {
    label: "Completed",
    className: "bg-green-100 text-green-700 border-green-200",
  },
};

interface StatusBadgeProps {
  status: Status;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const config = statusConfig[status] ?? statusConfig.draft;

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-semibold tracking-[0.02em]",
        config.className,
        className
      )}
      role="status"
    >
      {config.label}
    </span>
  );
}

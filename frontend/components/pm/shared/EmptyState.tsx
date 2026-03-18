// ============================================
// EmptyState — Strategy §5
// ============================================
// Centered icon + message + CTA button. Used for all empty state displays.
// Updated with premium animations and styling

import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  actionLabel?: string;
  actionHref?: string;
  onAction?: () => void;
  className?: string;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  actionLabel,
  actionHref,
  onAction,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center py-16 text-center",
        className
      )}
    >
      <div className="animate-fade-in" style={{ animation-delay: "0s" }}>
        <Icon className="h-6 w-6 text-muted-foreground mb-3" strokeWidth={1.75} />
      </div>
      <h3 className="text-base font-semibold text-foreground animate-fade-in" style={{ animation-delay: "0s" }}>{title}</h3>
      <p className="mt-2 max-w-sm text-sm text-muted-foreground animate-fade-in" style={{ animation-delay: "0s" }}>
        {description}
      </p>
      {actionLabel && actionHref && (
        <Link
          href={actionHref}
          className="mt-4 inline-flex items-center rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-all duration-200 hover:bg-primary/90 active:scale-[0.98] hover:shadow-lg hover:shadow-lg animate-fade-in"
          style={{
            boxShadow: "0 0 20px rgba(14, 116, 144, 0.3)",
          }}
        >
          {actionLabel} →
        </Link>
      )}
      {actionLabel && onAction && !actionHref && (
        <button
          onClick={onAction}
          className="mt-4 inline-flex items-center rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-all duration-200 hover:bg-primary/90 active:scale-[0.98] hover:shadow-lg hover:shadow-lg animate-fade-in"
          style={{
            boxShadow: "0 0 20px rgba(14, 116, 144, 0.3)",
          }}
        >
          {actionLabel} →
        </button>
      )}
    )
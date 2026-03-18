import { ReactNode } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  backLabel?: string;
  backHref?: string;
}

export default function PageHeader({
  title,
  description,
  actions,
  backLabel,
  backHref,
}: PageHeaderProps) {
  return (
    <div className="mb-8">
      {backLabel && backHref && (
        <Link
          href={backHref}
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors duration-150 mb-3"
        >
          <ArrowLeft className="h-4 w-4" strokeWidth={1.75} />
          {backLabel}
        </Link>
      )}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-[0.15em] text-muted-foreground">
            PM Pipeline
          </p>
          <h1 className="mt-2 font-[var(--font-display)] text-[1.875rem] font-semibold leading-[1.2] tracking-[-0.02em]">
            {title}
          </h1>
          {description && (
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
              description
            </p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}
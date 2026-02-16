"use client";

import { ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
}

export default function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div>
        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
          PM Pipeline
        </p>
        <h1 className="mt-2 text-3xl font-semibold">{title}</h1>
        {description && (
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

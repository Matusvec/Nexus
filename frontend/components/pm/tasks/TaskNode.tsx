// ============================================
// TaskNode — Strategy §4.11
// ============================================
// Collapsible single task node with effort badge and acceptance criteria.

"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { ScopeBadge, type ScopeSize } from "@/components/pm/shared/ScopeBadge";
import type { Task } from "@/lib/pm/types";
import { cn } from "@/lib/utils";

interface TaskNodeProps {
  task: Task;
  depth?: number;
}

export function TaskNode({ task, depth = 0 }: TaskNodeProps) {
  const [expanded, setExpanded] = useState(false);

  const hasContent =
    task.description ||
    (task.acceptance_criteria && task.acceptance_criteria.length > 0) ||
    (task.dependencies && task.dependencies.length > 0);

  return (
    <div className={cn("border-b border-border/50 last:border-b-0", depth > 0 && "ml-6")}>
      <div
        className={cn(
          "flex items-center gap-2 px-4 py-3 transition-colors duration-150",
          hasContent && "cursor-pointer hover:bg-muted/30"
        )}
        onClick={() => hasContent && setExpanded(!expanded)}
      >
        {hasContent ? (
          <button className="p-0.5 rounded hover:bg-muted transition-colors">
            {expanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
          </button>
        ) : (
          <div className="w-5" />
        )}

        <span className="flex-1 text-sm font-medium text-foreground">
          {task.title}
        </span>

        {task.estimated_effort && (
          <ScopeBadge scope={task.estimated_effort as ScopeSize} />
        )}
      </div>

      {/* Expanded content */}
      {expanded && hasContent && (
        <div className="px-4 pb-4 pl-11 space-y-3">
          {task.dependencies && task.dependencies.length > 0 && (
            <p className="text-xs text-muted-foreground">
              Depends on:{" "}
              {task.dependencies.join(", ")}
            </p>
          )}

          {task.description && (
            <p className="text-sm text-muted-foreground leading-relaxed">
              {task.description}
            </p>
          )}

          {task.acceptance_criteria && task.acceptance_criteria.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Acceptance Criteria
              </p>
              <ul className="space-y-1.5">
                {task.acceptance_criteria.map((criteria, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-foreground"
                  >
                    <span className="mt-0.5 h-4 w-4 shrink-0 rounded border border-border" />
                    {criteria}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Subtasks */}
      {task.subtasks &&
        task.subtasks.map((sub) => (
          <TaskNode key={sub.id} task={sub} depth={depth + 1} />
        ))}
    </div>
  );
}

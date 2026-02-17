// ============================================
// TaskTree — Strategy §4.11
// ============================================
// Full task tree with category tabs (Backend/Frontend/Data/QA).

"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ClipboardCopy, Download, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { TaskNode } from "./TaskNode";
import type { TaskTree as TaskTreeType, Task, TaskCategory } from "@/lib/pm/types";
import { cn } from "@/lib/utils";

interface TaskTreeProps {
  tree: TaskTreeType;
  onRegenerate?: () => void;
}

const categories: { key: TaskCategory; label: string }[] = [
  { key: "backend", label: "Backend" },
  { key: "frontend", label: "Frontend" },
  { key: "data", label: "Data" },
  { key: "qa", label: "QA" },
];

function tasksToMarkdown(tree: TaskTreeType): string {
  const lines: string[] = [
    `# Implementation Plan: ${tree.feature_name}`,
    ``,
    `${tree.total_tasks} tasks total`,
    ``,
  ];

  for (const cat of categories) {
    const tasks = tree[cat.key] as Task[];
    if (!tasks || tasks.length === 0) continue;

    lines.push(`## ${cat.label} (${tasks.length})`);
    lines.push(``);

    for (const task of tasks) {
      lines.push(`### ${task.title}${task.estimated_effort ? ` [${task.estimated_effort}]` : ""}`);
      if (task.description) lines.push(`\n${task.description}`);
      if (task.acceptance_criteria && task.acceptance_criteria.length > 0) {
        lines.push(`\n**Acceptance Criteria:**`);
        task.acceptance_criteria.forEach((c) => lines.push(`- [ ] ${c}`));
      }
      if (task.dependencies && task.dependencies.length > 0) {
        lines.push(`\n_Depends on: ${task.dependencies.join(", ")}_`);
      }
      lines.push(``);
    }
  }

  return lines.join("\n");
}

export function TaskTreeView({ tree, onRegenerate }: TaskTreeProps) {
  const [activeTab, setActiveTab] = useState<TaskCategory>("backend");

  const handleCopyMarkdown = async () => {
    const md = tasksToMarkdown(tree);
    try {
      await navigator.clipboard.writeText(md);
      toast.success("Copied to clipboard.");
    } catch {
      toast.error("Failed to copy.");
    }
  };

  const handleDownloadJSON = () => {
    const blob = new Blob([JSON.stringify(tree, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tasks-${tree.proposal_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Downloaded.");
  };

  return (
    <div className="space-y-6">
      <Tabs
        value={activeTab}
        onValueChange={(v) => setActiveTab(v as TaskCategory)}
      >
        <TabsList>
          {categories.map((cat) => {
            const tasks = tree[cat.key] as Task[];
            const count = tasks?.length ?? 0;
            return (
              <TabsTrigger key={cat.key} value={cat.key}>
                {cat.label} ({count})
              </TabsTrigger>
            );
          })}
        </TabsList>

        {categories.map((cat) => {
          const tasks = tree[cat.key] as Task[];
          return (
            <TabsContent key={cat.key} value={cat.key}>
              <div className="rounded-2xl border border-border bg-card overflow-hidden">
                {tasks && tasks.length > 0 ? (
                  tasks.map((task) => <TaskNode key={task.id} task={task} />)
                ) : (
                  <div className="py-8 text-center text-sm text-muted-foreground">
                    No {cat.label.toLowerCase()} tasks.
                  </div>
                )}
              </div>
            </TabsContent>
          );
        })}
      </Tabs>

      {/* Export actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleCopyMarkdown}
          className="inline-flex items-center gap-1.5 rounded-xl border border-border px-3 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted active:scale-[0.98]"
        >
          <ClipboardCopy className="h-4 w-4" strokeWidth={1.75} />
          Copy as Markdown
        </button>
        <button
          onClick={handleDownloadJSON}
          className="inline-flex items-center gap-1.5 rounded-xl border border-border px-3 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted active:scale-[0.98]"
        >
          <Download className="h-4 w-4" strokeWidth={1.75} />
          Download JSON
        </button>
        {onRegenerate && (
          <button
            onClick={onRegenerate}
            className="inline-flex items-center gap-1.5 rounded-xl border border-border px-3 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted active:scale-[0.98]"
          >
            <RefreshCw className="h-4 w-4" strokeWidth={1.75} />
            Regenerate
          </button>
        )}
      </div>
    </div>
  );
}

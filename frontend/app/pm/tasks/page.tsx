import PageHeader from "@/components/pm/PageHeader";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import { pmFetchSafe } from "@/lib/pm/api";
import type { RoadmapResponse, TaskTree } from "@/lib/pm/types";
import { ListChecks } from "lucide-react";
import Link from "next/link";

export default async function TasksPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = await searchParams;
  const proposalId =
    typeof sp.proposal === "string" ? sp.proposal : undefined;

  // If specific proposal requested, try to fetch its tasks
  let taskTree: TaskTree | null = null;
  if (proposalId) {
    taskTree = await pmFetchSafe<TaskTree>(
      `/feature_proposals/${proposalId}/tasks`
    );
  }

  // Get proposals that have tasks
  const roadmap = await pmFetchSafe<RoadmapResponse>("/roadmap");
  const proposals = roadmap?.items ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Tasks"
        description="Task trees grouped by proposal and category."
      />

      {/* Proposal selector */}
      {proposals.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {proposals.map((entry) => (
            <Link
              key={entry.proposal.id}
              href={`/pm/tasks?proposal=${entry.proposal.id}`}
              className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors duration-150 ${
                proposalId === entry.proposal.id
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:bg-muted"
              }`}
            >
              {entry.proposal.title}
            </Link>
          ))}
        </div>
      )}

      {!proposalId ? (
        <EmptyState
          icon={ListChecks}
          title="Select a proposal"
          description="Choose a proposal above to view its task tree, or generate tasks from a proposal detail page."
        />
      ) : !taskTree ? (
        <EmptyState
          icon={ListChecks}
          title="No tasks yet"
          description="Tasks haven&#39;t been generated for this proposal. Generate them from the proposal detail page."
          actionLabel="View Proposal"
          actionHref={`/pm/proposals/${proposalId}`}
        />
      ) : (
        <div className="space-y-6">
          {/* Summary */}
          <div className="rounded-2xl border border-border bg-card p-5">
            <h2 className="text-base font-semibold">
              {taskTree.feature_name}
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {taskTree.total_tasks} task
              {taskTree.total_tasks !== 1 ? "s" : ""} across{" "}
              {
                (
                  [
                    ["Backend", taskTree.backend],
                    ["Frontend", taskTree.frontend],
                    ["Data", taskTree.data],
                    ["QA", taskTree.qa],
                  ] as const
                ).filter(([, tasks]) => tasks.length > 0).length
              }{" "}
              categories
            </p>
          </div>

          {/* Task categories */}
          {(
            [
              ["Backend", taskTree.backend, "bg-blue-100 text-blue-700"],
              ["Frontend", taskTree.frontend, "bg-purple-100 text-purple-700"],
              ["Data", taskTree.data, "bg-amber-100 text-amber-700"],
              ["QA", taskTree.qa, "bg-green-100 text-green-700"],
            ] as const
          ).map(([label, tasks, badgeColor]) => {
            if (tasks.length === 0) return null;
            return (
              <div
                key={label}
                className="rounded-2xl border border-border bg-card p-5"
              >
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold">{label}</h3>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${badgeColor}`}
                  >
                    {tasks.length}
                  </span>
                </div>
                <div className="mt-4 space-y-2">
                  {tasks.map((task) => (
                    <div
                      key={task.id}
                      className="rounded-xl border border-border bg-muted/20 p-3"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-sm font-medium">{task.title}</p>
                        {task.estimated_effort && (
                          <span className="shrink-0 rounded-lg bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                            {task.estimated_effort}
                          </span>
                        )}
                      </div>
                      {task.description && (
                        <p className="mt-1 text-xs text-muted-foreground">
                          {task.description}
                        </p>
                      )}
                      {task.acceptance_criteria.length > 0 && (
                        <ul className="mt-2 space-y-1">
                          {task.acceptance_criteria.map((ac, i) => (
                            <li
                              key={i}
                              className="flex items-start gap-1.5 text-xs text-muted-foreground"
                            >
                              <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-muted-foreground/40" />
                              {ac}
                            </li>
                          ))}
                        </ul>
                      )}
                      {/* Subtasks */}
                      {task.subtasks.length > 0 && (
                        <div className="mt-3 ml-3 border-l border-border pl-3 space-y-2">
                          {task.subtasks.map((sub) => (
                            <div key={sub.id} className="text-xs">
                              <p className="font-medium text-foreground">
                                {sub.title}
                              </p>
                              {sub.description && (
                                <p className="mt-0.5 text-muted-foreground">
                                  {sub.description}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

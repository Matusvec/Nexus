import PageHeader from "@/components/pm/PageHeader";

export default function TasksPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Tasks"
        description="Task trees grouped by proposal and category."
      />
      <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
        Task trees will appear once task generation is enabled.
      </div>
    </div>
  );
}

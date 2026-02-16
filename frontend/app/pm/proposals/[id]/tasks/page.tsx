import PageHeader from "@/components/pm/PageHeader";

export default async function ProposalTasksPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="space-y-6">
      <PageHeader
        title="Task Tree"
        description={`Proposal ID: ${id}`}
      />
      <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
        Task tree view will appear once tasks are generated in the backend.
      </div>
    </div>
  );
}

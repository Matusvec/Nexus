import PageHeader from "@/components/pm/PageHeader";

export default function ProposalTasksPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Task Tree"
        description={`Proposal ID: ${params.id}`}
      />
      <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
        Task tree view will appear once tasks are generated in the backend.
      </div>
    </div>
  );
}

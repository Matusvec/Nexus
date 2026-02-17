import PageHeader from "@/components/pm/PageHeader";
import { EmptyState } from "@/components/pm/shared/EmptyState";
import { ClipboardList } from "lucide-react";

export default async function ProposalTasksPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="space-y-6">
      <PageHeader
        title="Task Tree"
        description={`Proposal ID: ${id}`}
        backLabel="Back to Proposal"
        backHref={`/pm/proposals/${id}`}
      />
      <EmptyState
        icon={ClipboardList}
        title="No tasks yet"
        description="Tasks will appear once they are generated from this proposal."
      />
    </div>
  );
}

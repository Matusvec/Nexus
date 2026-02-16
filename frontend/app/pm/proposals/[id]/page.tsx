import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";

export default async function ProposalDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="space-y-6">
      <PageHeader
        title="Proposal Detail"
        description={`Proposal ID: ${id}`}
      />
      <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
        Proposal detail rendering will be enabled when the proposal detail
        endpoint is implemented.
      </div>
      <Link href="/pm/proposals" className="text-sm text-primary underline">
        Back to Proposals
      </Link>
    </div>
  );
}

import PageHeader from "@/components/pm/PageHeader";
import Link from "next/link";

export default function ProposalDetailPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Proposal Detail"
        description={`Proposal ID: ${params.id}`}
      />
      <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
        Proposal detail rendering will be enabled when the proposal detail
        endpoint is implemented.
      </div>
      <Link href="/pm/proposals" className="text-sm text-primary underline">
        Back to Proposals
      </Link>
    </div>
  );
}

import PageHeader from "@/components/pm/PageHeader";

export default function ProposalsPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Proposals"
        description="Feature proposals generated from clusters, ready for review and approval."
      />
      <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
        Proposal list endpoint is not available yet. Generate proposals from a
        cluster once the backend proposal generator is wired.
      </div>
    </div>
  );
}

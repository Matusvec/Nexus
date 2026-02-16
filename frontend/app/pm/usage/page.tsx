import PageHeader from "@/components/pm/PageHeader";

export default function UsagePage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Usage"
        description="Track LLM cost, job history, and processing volume."
      />
      <div className="rounded-2xl border border-dashed border-border bg-white/60 p-8 text-center text-sm text-muted-foreground">
        Usage reporting will appear once cost tracking is wired in the backend.
      </div>
    </div>
  );
}

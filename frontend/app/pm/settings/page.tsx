import PageHeader from "@/components/pm/PageHeader";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Manage API keys, prompts, and pipeline configuration."
      />
      <div className="rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center text-sm text-muted-foreground">
        Settings UI is pending. This will surface API keys and prompt version
        controls.
      </div>
    </div>
  );
}

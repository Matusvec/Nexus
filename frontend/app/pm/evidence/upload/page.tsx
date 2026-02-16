"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import PageHeader from "@/components/pm/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { createEvidence, extractProblems } from "@/lib/pm/api";

const SOURCE_TYPES = [
  "interview",
  "support_ticket",
  "sales_note",
  "survey",
  "other",
] as const;

export default function EvidenceUploadPage() {
  const router = useRouter();
  const [form, setForm] = useState({
    title: "",
    source_type: "interview",
    persona: "",
    segment: "",
    source_date: "",
    raw_text: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onChange = (key: string, value: string) =>
    setForm((prev) => ({ ...prev, [key]: value }));

  const onSubmit = async () => {
    setIsSubmitting(true);
    setError(null);
    setStatus("Uploading evidence...");
    try {
      const evidence = await createEvidence({
        title: form.title,
        source_type: form.source_type,
        persona: form.persona || undefined,
        segment: form.segment || undefined,
        source_date: form.source_date || undefined,
        raw_text: form.raw_text,
      });

      setStatus("Triggering problem extraction...");
      try {
        await extractProblems(evidence.id);
        setStatus("Extraction queued. Redirecting...");
      } catch {
        // extraction trigger failed — still redirect, user can re-trigger later
        setStatus("Evidence saved. Extraction will be triggered separately.");
      }

      setTimeout(() => router.push("/pm/evidence"), 1200);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setStatus(null);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Upload Evidence"
        description="Paste a transcript or support log. We will chunk it, extract problems, and trace every claim back to a quote."
      />

      <div className="grid gap-6 lg:grid-cols-[2fr,1fr]">
        <div className="rounded-2xl border border-border bg-card/70 p-6">
          <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
            Raw text
          </label>
          <Textarea
            className="mt-3 min-h-[320px]"
            value={form.raw_text}
            onChange={(e) => onChange("raw_text", e.target.value)}
            placeholder="Paste interview transcript, support chat, or survey responses..."
          />
        </div>

        <div className="space-y-4 rounded-2xl border border-border bg-card/70 p-6">
          <div>
            <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
              Title
            </label>
            <Input
              className="mt-2"
              value={form.title}
              onChange={(e) => onChange("title", e.target.value)}
              placeholder="Customer Interview - Acme PM"
            />
          </div>
          <div>
            <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
              Source type
            </label>
            <select
              className="mt-2 w-full rounded-md border border-border bg-card px-3 py-2 text-sm"
              value={form.source_type}
              onChange={(e) => onChange("source_type", e.target.value)}
            >
              {SOURCE_TYPES.map((type) => (
                <option key={type} value={type}>
                  {type.replace("_", " ")}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
              Persona
            </label>
            <Input
              className="mt-2"
              value={form.persona}
              onChange={(e) => onChange("persona", e.target.value)}
              placeholder="Product Manager"
            />
          </div>
          <div>
            <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
              Segment
            </label>
            <Input
              className="mt-2"
              value={form.segment}
              onChange={(e) => onChange("segment", e.target.value)}
              placeholder="Enterprise"
            />
          </div>
          <div>
            <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
              Date
            </label>
            <Input
              className="mt-2"
              type="date"
              value={form.source_date}
              onChange={(e) => onChange("source_date", e.target.value)}
            />
          </div>

          {error && (
            <div className="rounded-xl border border-red-300 bg-red-500/10 px-3 py-2 text-sm text-red-400">
              {error}
            </div>
          )}

          {status && !error && (
            <div className="rounded-xl border border-blue-300 bg-blue-500/10 px-3 py-2 text-sm text-blue-400">
              {status}
            </div>
          )}

          <Button
            className="w-full"
            onClick={onSubmit}
            disabled={isSubmitting || !form.title || !form.raw_text}
          >
            {isSubmitting ? "Processing..." : "Upload & Process"}
          </Button>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import PageHeader from "@/components/pm/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

const SOURCE_TYPES = [
  "interview",
  "support_ticket",
  "sales_note",
  "survey",
  "other",
];

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
  const [error, setError] = useState<string | null>(null);

  const onChange = (key: string, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const onSubmit = async () => {
    setIsSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: form.title,
          source_type: form.source_type,
          persona: form.persona || undefined,
          segment: form.segment || undefined,
          source_date: form.source_date || undefined,
          raw_text: form.raw_text,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(data.detail || "Upload failed");
      }
      router.push("/pm/evidence");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
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
        <div className="rounded-2xl border border-border bg-white/70 p-6">
          <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
            Raw text
          </label>
          <Textarea
            className="mt-3 min-h-[320px]"
            value={form.raw_text}
            onChange={(event) => onChange("raw_text", event.target.value)}
            placeholder="Paste interview transcript, support chat, or survey responses..."
          />
        </div>

        <div className="space-y-4 rounded-2xl border border-border bg-white/70 p-6">
          <div>
            <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
              Title
            </label>
            <Input
              className="mt-2"
              value={form.title}
              onChange={(event) => onChange("title", event.target.value)}
              placeholder="Customer Interview - Acme PM"
            />
          </div>
          <div>
            <label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
              Source type
            </label>
            <select
              className="mt-2 w-full rounded-md border border-border bg-white px-3 py-2 text-sm"
              value={form.source_type}
              onChange={(event) => onChange("source_type", event.target.value)}
            >
              {SOURCE_TYPES.map((type) => (
                <option key={type} value={type}>
                  {type}
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
              onChange={(event) => onChange("persona", event.target.value)}
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
              onChange={(event) => onChange("segment", event.target.value)}
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
              onChange={(event) => onChange("source_date", event.target.value)}
            />
          </div>
          {error && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          <Button
            className="w-full"
            onClick={onSubmit}
            disabled={isSubmitting || !form.title || !form.raw_text}
          >
            {isSubmitting ? "Uploading..." : "Upload & Process"}
          </Button>
        </div>
      </div>
    </div>
  );
}

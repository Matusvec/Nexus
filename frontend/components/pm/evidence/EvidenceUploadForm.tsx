// ============================================
// EvidenceUploadForm — Strategy §4.3
// ============================================
// Upload form with react-dropzone + metadata fields + react-hook-form + zod.

"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { toast } from "sonner";
import { Upload, FileText, X } from "lucide-react";
import { createEvidence, extractProblems } from "@/lib/pm/api";
import { cn } from "@/lib/utils";

const evidenceSchema = z.object({
  title: z.string().min(1, "Title is required"),
  source_type: z.string().min(1, "Source type is required"),
  persona: z.string().optional(),
  segment: z.string().optional(),
  source_date: z.string().optional(),
  raw_text: z.string().min(1, "Text content is required"),
});

type EvidenceFormData = z.infer<typeof evidenceSchema>;

const sourceTypes = [
  { value: "interview", label: "Interview" },
  { value: "support_ticket", label: "Support Ticket" },
  { value: "sales_note", label: "Sales Note" },
  { value: "survey", label: "Survey" },
  { value: "other", label: "Other" },
];

export function EvidenceUploadForm() {
  const router = useRouter();
  const [droppedFile, setDroppedFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<EvidenceFormData>();

  const onDrop = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (!file) return;
      setDroppedFile(file);

      // Auto-populate title from filename
      const name = file.name.replace(/\.[^.]+$/, "");
      setValue("title", name);

      // Read text content
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        if (text) setValue("raw_text", text);
      };
      reader.readAsText(file);
    },
    [setValue]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/plain": [".txt"],
      "text/markdown": [".md"],
      "text/csv": [".csv"],
    },
    maxFiles: 1,
  });

  const onSubmit = async (data: EvidenceFormData) => {
    setSubmitting(true);
    try {
      const evidence = await createEvidence({
        title: data.title,
        source_type: data.source_type,
        persona: data.persona,
        segment: data.segment,
        source_date: data.source_date,
        raw_text: data.raw_text,
      });

      // Trigger extraction
      try {
        await extractProblems(evidence.id);
        toast.success("Evidence uploaded. Problem extraction started.");
      } catch {
        toast.success("Evidence uploaded. You can extract problems later.");
      }

      router.push("/pm/evidence");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="max-w-2xl space-y-6">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={cn(
          "flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 transition-colors duration-200 cursor-pointer",
          isDragActive
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/50"
        )}
      >
        <input {...getInputProps()} />
        {droppedFile ? (
          <div className="flex items-center gap-3">
            <FileText className="h-6 w-6 text-primary" strokeWidth={1.75} />
            <span className="text-sm font-medium">{droppedFile.name}</span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setDroppedFile(null);
                setValue("raw_text", "");
              }}
              className="rounded-full p-1 hover:bg-muted"
            >
              <X className="h-4 w-4 text-muted-foreground" />
            </button>
          </div>
        ) : (
          <>
            <Upload className="h-6 w-6 text-muted-foreground mb-2" strokeWidth={1.75} />
            <p className="text-sm text-muted-foreground">
              {isDragActive
                ? "Drop to upload"
                : "Drop a file here, or click to browse"}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Accepts: .txt, .md, .csv
            </p>
          </>
        )}
      </div>

      {/* OR divider */}
      <div className="flex items-center gap-4">
        <div className="h-px flex-1 bg-border" />
        <span className="text-xs text-muted-foreground uppercase tracking-wider">
          Or paste text directly
        </span>
        <div className="h-px flex-1 bg-border" />
      </div>

      {/* Text area */}
      <div className="space-y-1.5">
        <label htmlFor="raw_text" className="text-sm font-medium">
          Raw Text
        </label>
        <textarea
          id="raw_text"
          rows={8}
          {...register("raw_text")}
          className="w-full rounded-xl border border-input bg-[hsl(var(--input))] px-3 py-2 text-sm font-mono placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring resize-y"
          placeholder="Paste interview transcript, support ticket, or notes here..."
        />
        {errors.raw_text && (
          <p className="text-xs text-red-600">{errors.raw_text.message}</p>
        )}
      </div>

      {/* Metadata fields */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <label htmlFor="title" className="text-sm font-medium">
            Title *
          </label>
          <input
            id="title"
            {...register("title")}
            className="w-full rounded-xl border border-input bg-[hsl(var(--input))] px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder="Customer Interview - Acme Corp"
          />
          {errors.title && (
            <p className="text-xs text-red-600">{errors.title.message}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <label htmlFor="source_type" className="text-sm font-medium">
            Source Type *
          </label>
          <select
            id="source_type"
            {...register("source_type")}
            className="w-full rounded-xl border border-input bg-[hsl(var(--input))] px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="">Select type…</option>
            {sourceTypes.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
          {errors.source_type && (
            <p className="text-xs text-red-600">{errors.source_type.message}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <label htmlFor="persona" className="text-sm font-medium">
            Persona
          </label>
          <input
            id="persona"
            {...register("persona")}
            className="w-full rounded-xl border border-input bg-[hsl(var(--input))] px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder="Product Manager"
          />
        </div>

        <div className="space-y-1.5">
          <label htmlFor="segment" className="text-sm font-medium">
            Segment
          </label>
          <input
            id="segment"
            {...register("segment")}
            className="w-full rounded-xl border border-input bg-[hsl(var(--input))] px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder="Enterprise"
          />
        </div>

        <div className="space-y-1.5">
          <label htmlFor="source_date" className="text-sm font-medium">
            Date
          </label>
          <input
            id="source_date"
            type="date"
            {...register("source_date")}
            className="w-full rounded-xl border border-input bg-[hsl(var(--input))] px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end gap-3 pt-4">
        <button
          type="button"
          onClick={() => router.push("/pm/evidence")}
          className="rounded-xl border border-border px-4 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={submitting}
          className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/90 active:scale-[0.98] disabled:opacity-50"
        >
          {submitting ? "Uploading…" : "Upload & Extract →"}
        </button>
      </div>
    </form>
  );
}

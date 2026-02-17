"use client";

import PageHeader from "@/components/pm/PageHeader";
import { EvidenceUploadForm } from "@/components/pm/evidence/EvidenceUploadForm";

export default function EvidenceUploadPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Upload Evidence"
        description="Drag in a file or paste text. We&#39;ll chunk it, extract problems, and trace every claim back to a quote."
        backLabel="Back to Evidence"
        backHref="/pm/evidence"
      />
      <EvidenceUploadForm />
    </div>
  );
}

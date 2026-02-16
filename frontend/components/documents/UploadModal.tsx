"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  X,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useUIStore, useEvidenceStore, useJobsStore } from "@/lib/store";
import { createEvidence, extractProblems, getJobStatus } from "@/lib/api";
import type { SourceType, UploadProgress } from "@/lib/types";

const SOURCE_TYPES: { value: SourceType; label: string }[] = [
  { value: "interview", label: "Interview" },
  { value: "support_ticket", label: "Support Ticket" },
  { value: "sales_note", label: "Sales Note" },
  { value: "survey", label: "Survey" },
  { value: "other", label: "Other" },
];

export default function UploadModal() {
  const { isUploadModalOpen, setUploadModalOpen } = useUIStore();
  const { addItem } = useEvidenceStore();
  const { setJob } = useJobsStore();

  const [title, setTitle] = useState("");
  const [sourceType, setSourceType] = useState<SourceType>("interview");
  const [persona, setPersona] = useState("");
  const [segment, setSegment] = useState("");
  const [rawText, setRawText] = useState("");
  const [progress, setProgress] = useState<UploadProgress | null>(null);

  const reset = () => {
    setTitle("");
    setSourceType("interview");
    setPersona("");
    setSegment("");
    setRawText("");
    setProgress(null);
  };

  const handleSubmit = async () => {
    if (!title.trim() || !rawText.trim()) return;

    try {
      // Step 1: Create evidence
      setProgress({ stage: "submitting", progress: 30, message: "Creating evidence..." });
      const evidence = await createEvidence({
        title: title.trim(),
        source_type: sourceType,
        raw_text: rawText.trim(),
        persona: persona.trim() || undefined,
        segment: segment.trim() || undefined,
      });
      addItem(evidence);

      // Step 2: Start extraction job
      setProgress({ stage: "extracting", progress: 60, message: "Extracting problems..." });
      const job = await extractProblems(evidence.id);

      // Step 3: Poll job status
      const pollInterval = setInterval(async () => {
        try {
          const status = await getJobStatus(job.job_id);
          setJob(job.job_id, status);

          if (status.status === "completed") {
            clearInterval(pollInterval);
            setProgress({
              stage: "complete",
              progress: 100,
              message: `Done! Extracted ${status.result_count ?? 0} problems.`,
            });
          } else if (status.status === "failed") {
            clearInterval(pollInterval);
            setProgress({
              stage: "error",
              progress: 100,
              message: "Extraction failed",
              error: status.error ?? "Unknown error",
            });
          }
        } catch {
          clearInterval(pollInterval);
          setProgress({
            stage: "error",
            progress: 100,
            message: "Failed to check job status",
          });
        }
      }, 2000);
    } catch (err) {
      setProgress({
        stage: "error",
        progress: 0,
        message: "Submission failed",
        error: err instanceof Error ? err.message : "Unknown error",
      });
    }
  };

  const isSubmitting = progress !== null && progress.stage !== "complete" && progress.stage !== "error";
  const isDone = progress?.stage === "complete";

  return (
    <Dialog
      open={isUploadModalOpen}
      onOpenChange={(open) => {
        if (!open && !isSubmitting) {
          setUploadModalOpen(false);
          if (isDone) reset();
        }
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Add Evidence</DialogTitle>
          <DialogDescription>
            Paste an interview transcript, support ticket, sales note, or survey response.
          </DialogDescription>
        </DialogHeader>

        <AnimatePresence mode="wait">
          {!progress ? (
            <motion.div
              key="form"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              {/* Title */}
              <div>
                <label className="text-sm font-medium mb-1 block">Title *</label>
                <Input
                  placeholder="e.g. User interview — Jan 2026"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                />
              </div>

              {/* Source Type */}
              <div>
                <label className="text-sm font-medium mb-1 block">Source Type</label>
                <div className="flex flex-wrap gap-2">
                  {SOURCE_TYPES.map((st) => (
                    <Badge
                      key={st.value}
                      variant={sourceType === st.value ? "default" : "outline"}
                      className="cursor-pointer"
                      onClick={() => setSourceType(st.value)}
                    >
                      {st.label}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Optional fields */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-sm font-medium mb-1 block">Persona</label>
                  <Input
                    placeholder="e.g. Product Manager"
                    value={persona}
                    onChange={(e) => setPersona(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-1 block">Segment</label>
                  <Input
                    placeholder="e.g. Enterprise"
                    value={segment}
                    onChange={(e) => setSegment(e.target.value)}
                  />
                </div>
              </div>

              {/* Raw Text */}
              <div>
                <label className="text-sm font-medium mb-1 block">Evidence Text *</label>
                <Textarea
                  placeholder="Paste the full interview transcript, ticket body, or survey response here..."
                  value={rawText}
                  onChange={(e) => setRawText(e.target.value)}
                  rows={8}
                  className="resize-y"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  {rawText.length} characters
                </p>
              </div>

              {/* Submit */}
              <div className="flex justify-end gap-2 pt-2">
                <Button variant="outline" onClick={() => setUploadModalOpen(false)}>
                  Cancel
                </Button>
                <Button
                  onClick={handleSubmit}
                  disabled={!title.trim() || !rawText.trim()}
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Submit & Extract
                </Button>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="progress"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="py-8 text-center space-y-4"
            >
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-2 bg-primary/10">
                {progress.stage === "error" ? (
                  <AlertCircle className="w-8 h-8 text-destructive" />
                ) : progress.stage === "complete" ? (
                  <CheckCircle2 className="w-8 h-8 text-green-500" />
                ) : (
                  <Loader2 className="w-8 h-8 text-primary animate-spin" />
                )}
              </div>
              <p className="font-medium">{progress.message}</p>
              {progress.error && (
                <p className="text-sm text-destructive">{progress.error}</p>
              )}
              <Progress value={progress.progress} className="h-2 max-w-xs mx-auto" />

              {(isDone || progress.stage === "error") && (
                <Button
                  className="mt-4"
                  onClick={() => {
                    setUploadModalOpen(false);
                    reset();
                  }}
                >
                  {isDone ? "Done" : "Close"}
                </Button>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  );
}


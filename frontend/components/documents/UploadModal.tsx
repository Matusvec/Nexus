"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileText,
  X,
  CheckCircle2,
  AlertCircle,
  Loader2,
  File,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useUIStore, useDocumentsStore } from "@/lib/store";
import { formatBytes, generateId } from "@/lib/utils";
import type { UploadProgress } from "@/lib/types";

interface FileWithProgress {
  file: File;
  id: string;
  progress: UploadProgress;
}

const ACCEPTED_TYPES = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
    ".docx",
  ],
  "text/plain": [".txt"],
  "text/markdown": [".md"],
};

const stages: Array<{
  key: UploadProgress["stage"];
  label: string;
  progress: number;
}> = [
  { key: "parsing", label: "Parsing document...", progress: 25 },
  { key: "chunking", label: "Creating semantic chunks...", progress: 50 },
  { key: "embedding", label: "Generating embeddings...", progress: 75 },
  { key: "tree-building", label: "Building RAPTOR tree...", progress: 90 },
  { key: "complete", label: "Complete!", progress: 100 },
];

function FileUploadItem({
  fileWithProgress,
  onRemove,
}: {
  fileWithProgress: FileWithProgress;
  onRemove: () => void;
}) {
  const { file, progress } = fileWithProgress;
  const currentStage = stages.find((s) => s.key === progress.stage);
  const progressPercent = currentStage?.progress || 0;

  const getIcon = () => {
    if (progress.stage === "error") {
      return <AlertCircle className="w-5 h-5 text-destructive" />;
    }
    if (progress.stage === "complete") {
      return <CheckCircle2 className="w-5 h-5 text-green-500" />;
    }
    return <Loader2 className="w-5 h-5 text-primary animate-spin" />;
  };

  const getExtension = (name: string) => {
    const ext = name.split(".").pop()?.toUpperCase() || "FILE";
    return ext;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: -10 }}
      className="p-4 rounded-lg border border-border bg-card"
    >
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
          <FileText className="w-5 h-5 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <p className="font-medium text-sm truncate">{file.name}</p>
            <Badge variant="outline" className="text-xs flex-shrink-0">
              {getExtension(file.name)}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mb-2">
            {formatBytes(file.size)}
          </p>
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">{progress.message}</span>
              <span className="font-medium">{progressPercent}%</span>
            </div>
            <Progress value={progressPercent} className="h-1" />
          </div>
          {progress.stage === "error" && (
            <p className="text-xs text-destructive mt-2">{progress.error}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {getIcon()}
          {progress.stage !== "complete" && progress.stage !== "error" && (
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={onRemove}
            >
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export default function UploadModal() {
  const { isUploadModalOpen, setUploadModalOpen } = useUIStore();
  const { addDocument } = useDocumentsStore();
  const [files, setFiles] = useState<FileWithProgress[]>([]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: FileWithProgress[] = acceptedFiles.map((file) => ({
      file,
      id: generateId(),
      progress: {
        stage: "parsing",
        progress: 0,
        message: "Starting upload...",
      },
    }));

    setFiles((prev) => [...prev, ...newFiles]);

    // Simulate upload progress for each file
    newFiles.forEach((fileWithProgress) => {
      simulateUpload(fileWithProgress.id);
    });
  }, []);

  const simulateUpload = (fileId: string) => {
    const stageOrder: UploadProgress["stage"][] = [
      "parsing",
      "chunking",
      "embedding",
      "tree-building",
      "complete",
    ];

    let currentStageIndex = 0;

    const interval = setInterval(() => {
      currentStageIndex++;
      if (currentStageIndex >= stageOrder.length) {
        clearInterval(interval);
        return;
      }

      const stage = stageOrder[currentStageIndex];
      const stageInfo = stages.find((s) => s.key === stage);

      setFiles((prev) =>
        prev.map((f) =>
          f.id === fileId
            ? {
                ...f,
                progress: {
                  stage,
                  progress: stageInfo?.progress || 0,
                  message: stageInfo?.label || "",
                },
              }
            : f
        )
      );

      // When complete, add to documents store
      if (stage === "complete") {
        const fileWithProgress = files.find((f) => f.id === fileId);
        if (fileWithProgress) {
          addDocument({
            id: fileId,
            filename: fileWithProgress.file.name,
            uploadedAt: new Date().toISOString(),
            fileSize: fileWithProgress.file.size,
            chunkCount: Math.floor(Math.random() * 50) + 10,
            status: "ready",
          });
        }
      }
    }, 1500);
  };

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    multiple: true,
  });

  const completedCount = files.filter(
    (f) => f.progress.stage === "complete"
  ).length;
  const hasFiles = files.length > 0;

  return (
    <Dialog open={isUploadModalOpen} onOpenChange={setUploadModalOpen}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Upload Documents</DialogTitle>
          <DialogDescription>
            Add PDF, DOCX, TXT, or Markdown files to your knowledge base.
          </DialogDescription>
        </DialogHeader>

        {/* Dropzone */}
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
            transition-all duration-200
            ${
              isDragActive
                ? "border-primary bg-primary/5 scale-[1.02]"
                : "border-border hover:border-primary/50 hover:bg-muted/50"
            }
          `}
        >
          <input {...getInputProps()} />
          <motion.div
            animate={{ y: isDragActive ? -5 : 0 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
              <Upload
                className={`w-8 h-8 ${
                  isDragActive ? "text-primary" : "text-muted-foreground"
                }`}
              />
            </div>
            <p className="font-medium mb-1">
              {isDragActive ? "Drop files here" : "Drag & drop files here"}
            </p>
            <p className="text-sm text-muted-foreground mb-3">
              or click to browse
            </p>
            <div className="flex justify-center gap-2">
              <Badge variant="outline">PDF</Badge>
              <Badge variant="outline">DOCX</Badge>
              <Badge variant="outline">TXT</Badge>
              <Badge variant="outline">MD</Badge>
            </div>
          </motion.div>
        </div>

        {/* File List */}
        <AnimatePresence>
          {hasFiles && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
            >
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium">
                  Uploading {files.length} file{files.length !== 1 ? "s" : ""}
                </p>
                <Badge variant="success">
                  {completedCount}/{files.length} complete
                </Badge>
              </div>
              <ScrollArea className="max-h-[300px]">
                <div className="space-y-2">
                  {files.map((fileWithProgress) => (
                    <FileUploadItem
                      key={fileWithProgress.id}
                      fileWithProgress={fileWithProgress}
                      onRemove={() => removeFile(fileWithProgress.id)}
                    />
                  ))}
                </div>
              </ScrollArea>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <div className="flex justify-between items-center pt-4 border-t border-border">
          <p className="text-xs text-muted-foreground">
            Files are processed locally. Your data stays private.
          </p>
          <Button
            onClick={() => setUploadModalOpen(false)}
            disabled={files.some(
              (f) =>
                f.progress.stage !== "complete" && f.progress.stage !== "error"
            )}
          >
            {completedCount === files.length && hasFiles ? "Done" : "Close"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

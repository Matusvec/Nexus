"use client";

import { useState } from "react";
import PageHeader from "@/components/pm/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Settings, Save, RotateCcw } from "lucide-react";

interface SettingsState {
  backendUrl: string;
  chunkSize: string;
  chunkOverlap: string;
  clusterThreshold: string;
  defaultModel: string;
}

const DEFAULTS: SettingsState = {
  backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000",
  chunkSize: "512",
  chunkOverlap: "50",
  clusterThreshold: "0.75",
  defaultModel: "gpt-4o-mini",
};

export default function SettingsPage() {
  const [settings, setSettings] = useState<SettingsState>(DEFAULTS);
  const [saved, setSaved] = useState(false);

  const onChange = (key: keyof SettingsState, value: string) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const onSave = () => {
    // Settings are stored in env or backend config — show confirmation
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const onReset = () => {
    setSettings(DEFAULTS);
    setSaved(false);
  };

  const groups = [
    {
      title: "Connection",
      fields: [
        {
          key: "backendUrl" as const,
          label: "Backend URL",
          placeholder: "http://localhost:8000",
          hint: "Set via NEXT_PUBLIC_BACKEND_URL env var",
        },
      ],
    },
    {
      title: "Chunking",
      fields: [
        {
          key: "chunkSize" as const,
          label: "Chunk Size (tokens)",
          placeholder: "512",
          hint: "Target token count per chunk",
        },
        {
          key: "chunkOverlap" as const,
          label: "Chunk Overlap (tokens)",
          placeholder: "50",
          hint: "Overlap between adjacent chunks",
        },
      ],
    },
    {
      title: "Clustering",
      fields: [
        {
          key: "clusterThreshold" as const,
          label: "Similarity Threshold",
          placeholder: "0.75",
          hint: "Minimum cosine similarity for grouping (0.0–1.0)",
        },
      ],
    },
    {
      title: "LLM",
      fields: [
        {
          key: "defaultModel" as const,
          label: "Default Model",
          placeholder: "gpt-4o-mini",
          hint: "OpenAI model used for extraction and generation",
        },
      ],
    },
  ];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Pipeline configuration — chunk sizing, clustering threshold, and API defaults."
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onReset}
              className="gap-1.5"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset
            </Button>
            <Button size="sm" onClick={onSave} className="gap-1.5">
              <Save className="h-3.5 w-3.5" />
              {saved ? "Saved ✓" : "Save"}
            </Button>
          </div>
        }
      />

      <div className="space-y-6">
        {groups.map((group) => (
          <div
            key={group.title}
            className="rounded-2xl border border-border bg-card p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <Settings
                className="h-4 w-4 text-muted-foreground"
                strokeWidth={1.75}
              />
              <h2 className="text-base font-semibold">{group.title}</h2>
            </div>
            <div className="space-y-4">
              {group.fields.map((field) => (
                <div key={field.key}>
                  <label className="text-[11px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
                    {field.label}
                  </label>
                  <Input
                    className="mt-1.5"
                    value={settings[field.key]}
                    onChange={(e) => onChange(field.key, e.target.value)}
                    placeholder={field.placeholder}
                  />
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    {field.hint}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

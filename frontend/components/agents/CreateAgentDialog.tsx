"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Wrench, Plus, X } from "lucide-react";
import { createAgent, getAgentTools } from "@/lib/api";
import { useAgentsStore } from "@/lib/store";
import type { AgentTool } from "@/lib/types";
import { useEffect } from "react";

const DEFAULT_TOOLS = [
  "rag_search",
  "rag_tree_search",
  "document_list",
  "document_summary",
  "web_search",
  "youtube_search",
  "text_summarize",
  "calculate",
  "extract_entities",
];

export default function CreateAgentDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { addAgent } = useAgentsStore();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful AI agent. Answer questions accurately and use your tools when needed."
  );
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [temperature, setTemperature] = useState(0.7);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [availableTools, setAvailableTools] = useState<AgentTool[]>([]);

  useEffect(() => {
    if (open) {
      getAgentTools()
        .then(setAvailableTools)
        .catch(() => {
          // Use defaults if backend not available
          setAvailableTools(
            DEFAULT_TOOLS.map((t) => ({
              name: t,
              description: t.replace(/_/g, " "),
              parameters: [],
              category: "general",
            }))
          );
        });
    }
  }, [open]);

  const toggleTool = (name: string) => {
    setSelectedTools((prev) =>
      prev.includes(name) ? prev.filter((t) => t !== name) : [...prev, name]
    );
  };

  const handleCreate = async () => {
    if (!name.trim() || !systemPrompt.trim()) {
      setError("Name and system prompt are required");
      return;
    }

    setIsSubmitting(true);
    setError("");

    try {
      const newAgent = await createAgent({
        name: name.trim(),
        system_prompt: systemPrompt.trim(),
        description: description.trim(),
        tools: selectedTools.length > 0 ? selectedTools : undefined,
        temperature,
      });
      addAgent(newAgent);
      // Reset form
      setName("");
      setDescription("");
      setSystemPrompt(
        "You are a helpful AI agent. Answer questions accurately and use your tools when needed."
      );
      setSelectedTools([]);
      setTemperature(0.7);
      onClose();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to create agent. Make sure the backend is running."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Plus className="w-5 h-5" />
            Create Custom Agent
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Name */}
          <div>
            <Label htmlFor="agent-name">Agent Name</Label>
            <Input
              id="agent-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Math Tutor, Data Analyst..."
              className="mt-1"
            />
          </div>

          {/* Description */}
          <div>
            <Label htmlFor="agent-desc">Description</Label>
            <Input
              id="agent-desc"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Short description of what this agent does"
              className="mt-1"
            />
          </div>

          {/* System Prompt */}
          <div>
            <Label htmlFor="agent-prompt">System Prompt</Label>
            <Textarea
              id="agent-prompt"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Define the agent's behavior and personality..."
              className="mt-1 min-h-[100px]"
            />
          </div>

          {/* Tools */}
          <div>
            <Label>Tools</Label>
            <p className="text-xs text-muted-foreground mb-2">
              Select which tools this agent can use. Leave empty for all tools.
            </p>
            <div className="flex flex-wrap gap-2">
              {(availableTools.length > 0
                ? availableTools
                : DEFAULT_TOOLS.map((t) => ({
                    name: t,
                    description: t.replace(/_/g, " "),
                    parameters: [],
                    category: "general",
                  }))
              ).map((tool) => (
                <Badge
                  key={tool.name}
                  variant={
                    selectedTools.includes(tool.name) ? "default" : "outline"
                  }
                  className="cursor-pointer transition-colors"
                  onClick={() => toggleTool(tool.name)}
                >
                  <Wrench className="w-3 h-3 mr-1" />
                  {tool.name}
                </Badge>
              ))}
            </div>
          </div>

          {/* Temperature */}
          <div>
            <Label htmlFor="agent-temp">
              Temperature: {temperature.toFixed(1)}
            </Label>
            <input
              id="agent-temp"
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full mt-1"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Precise</span>
              <span>Creative</span>
            </div>
          </div>

          {/* Error */}
          {error && (
            <p className="text-sm text-red-500">{error}</p>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={isSubmitting}>
              {isSubmitting ? "Creating..." : "Create Agent"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Settings as SettingsIcon,
  Key,
  Cpu,
  Palette,
  Database,
  Check,
  Info,
} from "lucide-react";
import Sidebar from "@/components/layout/Sidebar";
import UploadModal from "@/components/documents/UploadModal";
import SearchCommand from "@/components/layout/SearchCommand";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

type SettingsSection = "general" | "models" | "api-keys" | "storage" | "appearance";

const sections: { id: SettingsSection; label: string; icon: typeof SettingsIcon; description: string }[] = [
  {
    id: "general",
    label: "General",
    icon: SettingsIcon,
    description: "App preferences and behavior",
  },
  {
    id: "models",
    label: "Models & Runtime",
    icon: Cpu,
    description: "LLM model selection and local runtime",
  },
  {
    id: "api-keys",
    label: "API Keys",
    icon: Key,
    description: "Configure provider API keys",
  },
  {
    id: "storage",
    label: "Storage & Data",
    icon: Database,
    description: "Vector store and document storage",
  },
  {
    id: "appearance",
    label: "Appearance",
    icon: Palette,
    description: "Theme and visual preferences",
  },
];

function SettingsField({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-6 py-4">
      <div className="flex-1">
        <p className="text-sm font-medium">{label}</p>
        {description && (
          <p className="text-sm text-muted-foreground mt-0.5">{description}</p>
        )}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  );
}

function GeneralSettings() {
  return (
    <div className="space-y-1">
      <SettingsField
        label="Default Persona"
        description="The AI persona selected by default in new conversations"
      >
        <select className="h-9 rounded-md border border-input bg-background px-3 text-sm">
          <option>Max — Mechanical Engineer</option>
          <option>Dr. Elena — Physicist</option>
          <option>Byte — Software Engineer</option>
          <option>Stacy — Electrical Engineer</option>
        </select>
      </SettingsField>
      <Separator />
      <SettingsField
        label="Auto-build RAPTOR tree"
        description="Automatically build the hierarchical tree after document upload"
      >
        <Button variant="outline" size="sm" className="gap-2">
          <Check className="w-4 h-4 text-green-400" />
          Enabled
        </Button>
      </SettingsField>
      <Separator />
      <SettingsField
        label="Chat history"
        description="Persist conversation history between sessions"
      >
        <Button variant="outline" size="sm" className="gap-2">
          <Check className="w-4 h-4 text-green-400" />
          Enabled
        </Button>
      </SettingsField>
      <Separator />
      <SettingsField
        label="Retrieval top-K"
        description="Default number of chunks returned per query"
      >
        <Input
          type="number"
          defaultValue="10"
          className="w-20 h-9 text-center"
        />
      </SettingsField>
    </div>
  );
}

function ModelsSettings() {
  const models = [
    {
      name: "Gemini 1.5 Flash",
      provider: "Google",
      status: "active",
      description: "Fast and efficient for most queries",
    },
    {
      name: "Gemini 1.5 Pro",
      provider: "Google",
      status: "available",
      description: "Higher quality for complex reasoning",
    },
    {
      name: "Local LLM (Ollama)",
      provider: "Local",
      status: "not-configured",
      description: "Run models locally for full privacy",
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
        <Info className="w-4 h-4 text-blue-400 flex-shrink-0" />
        <p className="text-sm text-blue-300">
          Nexus supports multiple LLM providers. Configure your preferred model below.
        </p>
      </div>

      <div className="space-y-3">
        {models.map((model) => (
          <Card
            key={model.name}
            className={cn(
              "transition-all",
              model.status === "active" && "border-primary"
            )}
          >
            <CardContent className="p-4 flex items-center gap-4">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <Cpu className="w-5 h-5 text-primary" />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <p className="font-medium text-sm">{model.name}</p>
                  <Badge variant="outline" className="text-xs">
                    {model.provider}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  {model.description}
                </p>
              </div>
              {model.status === "active" ? (
                <Badge variant="success" className="text-xs">
                  Active
                </Badge>
              ) : model.status === "available" ? (
                <Button variant="outline" size="sm">
                  Select
                </Button>
              ) : (
                <Button variant="outline" size="sm">
                  Configure
                </Button>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      <Separator />

      <SettingsField
        label="Embedding Model"
        description="Model used for generating document chunk embeddings"
      >
        <select className="h-9 rounded-md border border-input bg-background px-3 text-sm">
          <option>all-MiniLM-L6-v2</option>
          <option>text-embedding-3-small</option>
        </select>
      </SettingsField>
    </div>
  );
}

function ApiKeysSettings() {
  const keys = [
    {
      name: "Google AI (Gemini)",
      envVar: "GOOGLE_API_KEY",
      configured: true,
    },
    {
      name: "OpenAI",
      envVar: "OPENAI_API_KEY",
      configured: false,
    },
    {
      name: "Anthropic",
      envVar: "ANTHROPIC_API_KEY",
      configured: false,
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
        <Info className="w-4 h-4 text-yellow-400 flex-shrink-0" />
        <p className="text-sm text-yellow-300">
          API keys are stored locally and never sent to third-party services. Set
          them in your <code className="font-mono">.env.local</code> file.
        </p>
      </div>

      <div className="space-y-1">
        {keys.map((key) => (
          <div key={key.name}>
            <SettingsField label={key.name} description={`Environment: ${key.envVar}`}>
              <div className="flex items-center gap-2">
                {key.configured ? (
                  <Badge variant="success" className="text-xs">
                    <Check className="w-3 h-3 mr-1" />
                    Configured
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-xs">
                    Not set
                  </Badge>
                )}
              </div>
            </SettingsField>
            <Separator />
          </div>
        ))}
      </div>
    </div>
  );
}

function StorageSettings() {
  return (
    <div className="space-y-4">
      <SettingsField
        label="Vector Database"
        description="ChromaDB is used as the local vector store"
      >
        <Badge variant="success" className="text-xs">
          <Database className="w-3 h-3 mr-1" />
          Connected
        </Badge>
      </SettingsField>
      <Separator />
      <SettingsField
        label="Storage Location"
        description="Where documents and embeddings are stored locally"
      >
        <code className="text-xs font-mono bg-muted px-2 py-1 rounded">
          ./backend/chroma_db/
        </code>
      </SettingsField>
      <Separator />
      <SettingsField
        label="Clear All Data"
        description="Delete all documents, embeddings, and conversation history"
      >
        <Button variant="destructive" size="sm">
          Clear Data
        </Button>
      </SettingsField>
    </div>
  );
}

function AppearanceSettings() {
  return (
    <div className="space-y-4">
      <SettingsField
        label="Theme"
        description="Nexus uses a dark theme optimized for focus"
      >
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" className="gap-2">
            <div className="w-4 h-4 rounded-full bg-background border border-border" />
            Dark
          </Button>
          <Button variant="outline" size="sm" className="gap-2" disabled>
            <div className="w-4 h-4 rounded-full bg-white border" />
            Light
            <Badge variant="outline" className="text-xs ml-1">
              Soon
            </Badge>
          </Button>
        </div>
      </SettingsField>
      <Separator />
      <SettingsField
        label="Accent Color"
        description="Primary color used across the interface"
      >
        <div className="flex gap-2">
          {["#3B82F6", "#8B5CF6", "#06B6D4", "#10B981", "#F97316"].map(
            (color) => (
              <button
                key={color}
                className={cn(
                  "w-8 h-8 rounded-lg transition-all hover:scale-110",
                  color === "#3B82F6" && "ring-2 ring-offset-2 ring-offset-background ring-white"
                )}
                style={{ backgroundColor: color }}
                aria-label={`Select accent color ${color}`}
              />
            )
          )}
        </div>
      </SettingsField>
      <Separator />
      <SettingsField
        label="Sidebar Default"
        description="Start with sidebar expanded or collapsed"
      >
        <select className="h-9 rounded-md border border-input bg-background px-3 text-sm">
          <option>Expanded</option>
          <option>Collapsed</option>
        </select>
      </SettingsField>
    </div>
  );
}

export default function SettingsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeSection, setActiveSection] = useState<SettingsSection>("general");

  const renderSection = () => {
    switch (activeSection) {
      case "general":
        return <GeneralSettings />;
      case "models":
        return <ModelsSettings />;
      case "api-keys":
        return <ApiKeysSettings />;
      case "storage":
        return <StorageSettings />;
      case "appearance":
        return <AppearanceSettings />;
    }
  };

  const activeSectionData = sections.find((s) => s.id === activeSection)!;

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 border-b border-border flex items-center px-6">
          <div>
            <h1 className="text-xl font-semibold">Settings</h1>
            <p className="text-sm text-muted-foreground">
              Configure Nexus to your preferences
            </p>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Section Nav */}
          <nav className="w-64 border-r border-border p-4 space-y-1">
            {sections.map((section) => {
              const isActive = activeSection === section.id;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "hover:bg-muted text-muted-foreground hover:text-foreground"
                  )}
                >
                  <section.icon className="w-4 h-4 flex-shrink-0" />
                  <span className="text-sm font-medium">{section.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Section Content */}
          <ScrollArea className="flex-1 p-6">
            <motion.div
              key={activeSection}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="max-w-2xl"
            >
              <div className="mb-6">
                <h2 className="text-lg font-semibold">{activeSectionData.label}</h2>
                <p className="text-sm text-muted-foreground">
                  {activeSectionData.description}
                </p>
              </div>
              {renderSection()}
            </motion.div>
          </ScrollArea>
        </div>
      </main>

      <UploadModal />
      <SearchCommand />
    </div>
  );
}

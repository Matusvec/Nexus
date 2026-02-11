"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bot,
  Plus,
  Play,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  History,
  Zap,
  FileText,
  Search,
  MessageSquare,
  Hand,
  ChevronRight,
} from "lucide-react";
import Sidebar from "@/components/layout/Sidebar";
import UploadModal from "@/components/documents/UploadModal";
import SearchCommand from "@/components/layout/SearchCommand";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { PERSONAS, type PersonaId } from "@/lib/types";
import { cn } from "@/lib/utils";

// Mock agent run data
interface AgentRun {
  id: string;
  agentId: PersonaId;
  input: string;
  output?: string;
  status: "running" | "completed" | "failed";
  toolCalls: { tool: string; input: string; output: string }[];
  startedAt: string;
  completedAt?: string;
}

const mockRuns: AgentRun[] = [
  {
    id: "run-1",
    agentId: "max",
    input: "Will this motor mount handle 5G acceleration?",
    output:
      "Based on the Motor_Specifications_v2.pdf, the brushless DC motor with 7075 aluminum brackets has a 3x safety margin at 5G acceleration. The key limiting factor is the bearing fatigue life at sustained loads.",
    status: "completed",
    toolCalls: [
      {
        tool: "query_group",
        input: 'query="motor mount 5G acceleration", group="Mechanical Engineering"',
        output: "Found 3 relevant chunks from Motor_Specifications_v2.pdf",
      },
      {
        tool: "search_all_groups",
        input: 'query="aluminum 7075 fatigue"',
        output: "Found 1 result in Electronics group (cross-reference)",
      },
    ],
    startedAt: "2026-02-10T14:30:00Z",
    completedAt: "2026-02-10T14:30:12Z",
  },
  {
    id: "run-2",
    agentId: "elena",
    input: "Explain the tunneling probability in the quantum computing paper",
    output:
      "The tunneling probability depends primarily on the barrier width and height. According to the Quantum_Computing_Intro.pdf, for a rectangular barrier of width d and height V\u2080...",
    status: "completed",
    toolCalls: [
      {
        tool: "query_group",
        input: 'query="tunneling probability barrier", group="Physics Research"',
        output: "Found 2 relevant chunks at Layer 0",
      },
    ],
    startedAt: "2026-02-10T12:15:00Z",
    completedAt: "2026-02-10T12:15:08Z",
  },
  {
    id: "run-3",
    agentId: "byte",
    input: "Refactor the embedding pipeline to use batch processing",
    status: "running",
    toolCalls: [
      {
        tool: "query_group",
        input: 'query="embedding pipeline implementation", group="Software & Algorithms"',
        output: "Found 4 chunks from Implementation_Notes.md",
      },
    ],
    startedAt: "2026-02-10T16:00:00Z",
  },
];

const agentTools = [
  {
    name: "query_group",
    description: "Search within a specific document group using RAPTOR retrieval",
    icon: Search,
  },
  {
    name: "search_all_groups",
    description: "Search across all document groups",
    icon: FileText,
  },
  {
    name: "get_connected_groups",
    description: "Navigate connections in the knowledge mind-map",
    icon: Zap,
  },
  {
    name: "request_human_task",
    description: "Delegate physical tasks to the user with instructions",
    icon: Hand,
  },
  {
    name: "suggest_connection",
    description: "Propose new connections between document groups",
    icon: MessageSquare,
  },
];

function AgentCard({
  personaId,
  isSelected,
  onClick,
}: {
  personaId: PersonaId;
  isSelected: boolean;
  onClick: () => void;
}) {
  const persona = PERSONAS[personaId];
  const runCount = mockRuns.filter((r) => r.agentId === personaId).length;

  return (
    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
      <Card
        className={cn(
          "cursor-pointer transition-all",
          isSelected
            ? "border-primary shadow-lg shadow-primary/10"
            : "hover:border-primary/30"
        )}
        onClick={onClick}
      >
        <CardContent className="p-5">
          <div className="flex items-start gap-4">
            <div
              className="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl flex-shrink-0"
              style={{ backgroundColor: `${persona.color}20` }}
            >
              {persona.avatar}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="font-semibold">{persona.name}</h3>
                <Badge
                  variant="outline"
                  className="text-xs"
                  style={{ borderColor: `${persona.color}50`, color: persona.color }}
                >
                  {persona.role}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mb-3">
                {persona.description}
              </p>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Play className="w-3 h-3" />
                  {runCount} run{runCount !== 1 ? "s" : ""}
                </span>
                <span className="flex items-center gap-1">
                  <Zap className="w-3 h-3" />
                  {agentTools.length} tools
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function RunHistoryItem({ run }: { run: AgentRun }) {
  const persona = PERSONAS[run.agentId];
  const [expanded, setExpanded] = useState(false);

  const statusConfig = {
    running: {
      icon: Loader2,
      className: "text-blue-400 animate-spin",
      badge: "info" as const,
    },
    completed: {
      icon: CheckCircle2,
      className: "text-green-400",
      badge: "success" as const,
    },
    failed: {
      icon: AlertCircle,
      className: "text-red-400",
      badge: "destructive" as const,
    },
  };

  const config = statusConfig[run.status];
  const StatusIcon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="border border-border rounded-xl overflow-hidden"
    >
      <button
        className="w-full p-4 flex items-start gap-3 cursor-pointer hover:bg-muted/30 transition-colors text-left"
        onClick={() => setExpanded(!expanded)}
      >
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center text-sm flex-shrink-0"
          style={{ backgroundColor: `${persona.color}20` }}
        >
          {persona.avatar}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium" style={{ color: persona.color }}>
              {persona.name}
            </span>
            <Badge variant={config.badge} className="text-xs">
              {run.status}
            </Badge>
          </div>
          <p className="text-sm truncate">{run.input}</p>
          <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {new Date(run.startedAt).toLocaleString()}
            </span>
            <span>{run.toolCalls.length} tool call{run.toolCalls.length !== 1 ? "s" : ""}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusIcon className={cn("w-4 h-4", config.className)} />
          <ChevronRight
            className={cn(
              "w-4 h-4 text-muted-foreground transition-transform",
              expanded && "rotate-90"
            )}
          />
        </div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-border"
          >
            <div className="p-4 space-y-4">
              {/* Tool Calls */}
              <div>
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                  Tool Calls
                </p>
                <div className="space-y-2">
                  {run.toolCalls.map((tc, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-lg bg-muted/50 text-sm"
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <Zap className="w-3 h-3 text-primary" />
                        <code className="text-xs font-mono text-primary">
                          {tc.tool}
                        </code>
                      </div>
                      <p className="text-xs text-muted-foreground font-mono">
                        {tc.input}
                      </p>
                      <Separator className="my-2" />
                      <p className="text-xs">{tc.output}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Output */}
              {run.output && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                    Output
                  </p>
                  <div className="p-3 rounded-lg bg-muted/50 text-sm leading-relaxed">
                    {run.output}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default function AgentsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<PersonaId | null>(null);
  const [activeTab, setActiveTab] = useState("library");

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 border-b border-border flex items-center justify-between px-6">
          <div>
            <h1 className="text-xl font-semibold">Agents</h1>
            <p className="text-sm text-muted-foreground">
              AI specialist personas and their run history
            </p>
          </div>
          <Button variant="outline" size="sm">
            <Plus className="w-4 h-4 mr-2" />
            Create Agent
          </Button>
        </header>

        {/* Tabs */}
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          className="flex-1 flex flex-col overflow-hidden"
        >
          <div className="px-6 pt-4 border-b border-border">
            <TabsList>
              <TabsTrigger value="library">
                <Bot className="w-4 h-4 mr-2" />
                Agent Library
              </TabsTrigger>
              <TabsTrigger value="history">
                <History className="w-4 h-4 mr-2" />
                Run History
              </TabsTrigger>
              <TabsTrigger value="tools">
                <Zap className="w-4 h-4 mr-2" />
                Available Tools
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Agent Library */}
          <TabsContent value="library" className="flex-1 overflow-auto p-6 mt-0">
            <div className="grid md:grid-cols-2 gap-4 max-w-4xl">
              {(Object.keys(PERSONAS) as PersonaId[]).map((id) => (
                <AgentCard
                  key={id}
                  personaId={id}
                  isSelected={selectedAgent === id}
                  onClick={() =>
                    setSelectedAgent(selectedAgent === id ? null : id)
                  }
                />
              ))}
            </div>

            {/* Create custom agent prompt */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mt-6 max-w-4xl"
            >
              <Card className="border-dashed border-2 border-border hover:border-primary/30 transition-colors cursor-pointer">
                <CardContent className="p-6 flex items-center gap-4">
                  <div className="w-14 h-14 rounded-2xl bg-muted flex items-center justify-center">
                    <Plus className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Create Custom Agent</h3>
                    <p className="text-sm text-muted-foreground">
                      Define a new specialist with custom personality, expertise,
                      and tool access
                    </p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Run History */}
          <TabsContent value="history" className="flex-1 overflow-auto p-6 mt-0">
            <div className="max-w-3xl space-y-3">
              {mockRuns.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-center py-20"
                >
                  <div className="w-20 h-20 rounded-2xl bg-muted flex items-center justify-center mx-auto mb-6">
                    <History className="w-10 h-10 text-muted-foreground" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">No runs yet</h3>
                  <p className="text-muted-foreground max-w-md mx-auto">
                    Select an agent and start a conversation. All agent runs and
                    tool calls will appear here.
                  </p>
                </motion.div>
              ) : (
                mockRuns.map((run) => (
                  <RunHistoryItem key={run.id} run={run} />
                ))
              )}
            </div>
          </TabsContent>

          {/* Tools */}
          <TabsContent value="tools" className="flex-1 overflow-auto p-6 mt-0">
            <div className="max-w-3xl space-y-3">
              {agentTools.map((tool, idx) => (
                <motion.div
                  key={tool.name}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                >
                  <Card>
                    <CardContent className="p-4 flex items-center gap-4">
                      <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                        <tool.icon className="w-5 h-5 text-primary" />
                      </div>
                      <div className="flex-1">
                        <code className="text-sm font-mono font-medium">
                          {tool.name}
                        </code>
                        <p className="text-sm text-muted-foreground mt-0.5">
                          {tool.description}
                        </p>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        Available
                      </Badge>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </main>

      <UploadModal />
      <SearchCommand />
    </div>
  );
}

"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bot,
  Plus,
  Trash2,
  MessageSquare,
  Settings,
  Users,
  Wrench,
  Search,
  Code,
  Globe,
  FileText,
  Zap,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  type AgentInfo,
  type AgentRoleType,
  AGENT_ROLES,
} from "@/lib/types";
import { useAgentsStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import CreateAgentDialog from "./CreateAgentDialog";
import AgentChat from "./AgentChat";
import OrchestratorPanel from "./OrchestratorPanel";

const roleIcons: Record<AgentRoleType, React.ReactNode> = {
  research: <Search className="w-5 h-5" />,
  code: <Code className="w-5 h-5" />,
  web_search: <Globe className="w-5 h-5" />,
  document: <FileText className="w-5 h-5" />,
  custom: <Settings className="w-5 h-5" />,
  orchestrator: <Users className="w-5 h-5" />,
};

export default function AgentsPanel() {
  const {
    agents,
    selectedAgentId,
    selectAgent,
    isCreateAgentOpen,
    setCreateAgentOpen,
  } = useAgentsStore();

  const [activeTab, setActiveTab] = useState<"agents" | "orchestrator">(
    "agents"
  );

  const selectedAgent = agents.find((a) => a.id === selectedAgentId);

  return (
    <div className="flex h-full">
      {/* Sidebar - Agent List */}
      <div className="w-80 border-r border-border flex flex-col bg-card/50">
        {/* Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Bot className="w-5 h-5 text-primary" />
              <h2 className="font-semibold">AI Agents</h2>
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setCreateAgentOpen(true)}
            >
              <Plus className="w-4 h-4 mr-1" />
              New
            </Button>
          </div>

          {/* Tab Switcher */}
          <div className="flex gap-1 bg-muted rounded-lg p-1">
            <button
              className={cn(
                "flex-1 text-sm py-1.5 rounded-md transition-colors",
                activeTab === "agents"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
              onClick={() => setActiveTab("agents")}
            >
              Agents
            </button>
            <button
              className={cn(
                "flex-1 text-sm py-1.5 rounded-md transition-colors",
                activeTab === "orchestrator"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
              onClick={() => setActiveTab("orchestrator")}
            >
              Orchestrator
            </button>
          </div>
        </div>

        {activeTab === "agents" ? (
          <ScrollArea className="flex-1">
            <div className="p-2 space-y-1">
              {agents.map((agent) => {
                const roleConfig =
                  AGENT_ROLES[agent.config.role] || AGENT_ROLES.custom;
                return (
                  <motion.button
                    key={agent.id}
                    className={cn(
                      "w-full text-left p-3 rounded-lg transition-colors",
                      selectedAgentId === agent.id
                        ? "bg-primary/10 border border-primary/20"
                        : "hover:bg-muted/50"
                    )}
                    onClick={() => selectAgent(agent.id)}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center text-xl"
                        style={{
                          backgroundColor: `${roleConfig.color}20`,
                        }}
                      >
                        {roleConfig.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-medium text-sm truncate">
                            {agent.config.name}
                          </p>
                          {agent.is_custom && (
                            <Badge variant="outline" className="text-xs">
                              Custom
                            </Badge>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground truncate">
                          {agent.config.description || roleConfig.label}
                        </p>
                      </div>
                      <ChevronRight className="w-4 h-4 text-muted-foreground" />
                    </div>
                  </motion.button>
                );
              })}

              {agents.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <Bot className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No agents loaded</p>
                  <p className="text-xs">
                    Connect to the backend to see agents
                  </p>
                </div>
              )}
            </div>
          </ScrollArea>
        ) : (
          <OrchestratorPanel agents={agents} />
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {selectedAgent && activeTab === "agents" ? (
          <AgentChat agent={selectedAgent} />
        ) : activeTab === "orchestrator" ? (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Users className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="font-medium">Orchestrator Workspace</p>
              <p className="text-sm mt-1">
                Use the panel on the left to start an orchestrator session
              </p>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Bot className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="font-medium">Select an Agent</p>
              <p className="text-sm mt-1">
                Choose an agent from the list to start chatting
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Create Agent Dialog */}
      <CreateAgentDialog
        open={isCreateAgentOpen}
        onClose={() => setCreateAgentOpen(false)}
      />
    </div>
  );
}

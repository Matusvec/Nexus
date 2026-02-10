"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Users,
  Plus,
  MessageSquare,
  Send,
  Loader2,
  Bot,
  User,
  ArrowRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  type AgentInfo,
  type OrchestratorMessage,
  AGENT_ROLES,
} from "@/lib/types";
import {
  createOrchestratorSession,
  sendOrchestratorMessage,
} from "@/lib/api";
import { useAgentsStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export default function OrchestratorPanel({
  agents,
}: {
  agents: AgentInfo[];
}) {
  const {
    orchestratorSessionId,
    orchestratorMessages,
    setOrchestratorSessionId,
    setOrchestratorMessages,
    addOrchestratorMessage,
  } = useAgentsStore();

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionName, setSessionName] = useState("New Session");

  const handleCreateSession = async () => {
    try {
      const session = await createOrchestratorSession(sessionName);
      setOrchestratorSessionId(session.id);
      setOrchestratorMessages([]);
    } catch {
      // If backend is down, create a local-only session
      setOrchestratorSessionId("local-session");
      setOrchestratorMessages([]);
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isLoading || !orchestratorSessionId) return;

    setInput("");

    // Add user message locally
    const userMsg: OrchestratorMessage = {
      id: `user-${Date.now()}`,
      sender: "user",
      sender_name: "User",
      content: text,
      timestamp: Date.now() / 1000,
      message_type: "message",
      metadata: {},
    };
    addOrchestratorMessage(userMsg);
    setIsLoading(true);

    try {
      const result = await sendOrchestratorMessage(
        orchestratorSessionId,
        text
      );

      // Update messages from session
      if (result.session_messages) {
        setOrchestratorMessages(result.session_messages);
      }
    } catch (err) {
      const errorMsg: OrchestratorMessage = {
        id: `error-${Date.now()}`,
        sender: "orchestrator",
        sender_name: "Orchestrator",
        content: `Error: ${err instanceof Error ? err.message : "Failed to reach orchestrator. Make sure the backend is running."}`,
        timestamp: Date.now() / 1000,
        message_type: "message",
        metadata: {},
      };
      addOrchestratorMessage(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const getSenderColor = (sender: string): string => {
    if (sender === "user") return "#6366F1";
    if (sender === "orchestrator") return "#EC4899";
    const agent = agents.find((a) => a.id === sender);
    if (agent) {
      const roleConfig = AGENT_ROLES[agent.config.role];
      return roleConfig?.color || "#6B7280";
    }
    return "#6B7280";
  };

  const getSenderIcon = (sender: string): string => {
    if (sender === "user") return "👤";
    if (sender === "orchestrator") return "🎯";
    const agent = agents.find((a) => a.id === sender);
    if (agent) {
      const roleConfig = AGENT_ROLES[agent.config.role];
      return roleConfig?.icon || "🤖";
    }
    return "🤖";
  };

  if (!orchestratorSessionId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6">
        <Users className="w-12 h-12 text-muted-foreground/50 mb-4" />
        <h3 className="font-semibold mb-2">Orchestrator</h3>
        <p className="text-sm text-muted-foreground text-center mb-4">
          Create a session where agents collaborate to solve complex tasks
        </p>
        <Input
          value={sessionName}
          onChange={(e) => setSessionName(e.target.value)}
          placeholder="Session name"
          className="mb-3 max-w-[200px]"
        />
        <Button onClick={handleCreateSession} size="sm">
          <Plus className="w-4 h-4 mr-1" />
          Start Session
        </Button>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Session Header */}
      <div className="p-3 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">Session Active</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="text-xs"
            onClick={() => {
              setOrchestratorSessionId(null);
              setOrchestratorMessages([]);
            }}
          >
            End
          </Button>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-3">
        <div className="space-y-3">
          {orchestratorMessages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-2"
            >
              <div
                className="w-6 h-6 rounded-md flex items-center justify-center text-xs flex-shrink-0 mt-0.5"
                style={{
                  backgroundColor: `${getSenderColor(msg.sender)}20`,
                }}
              >
                {getSenderIcon(msg.sender)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-medium">
                    {msg.sender_name}
                  </span>
                  {msg.message_type !== "message" && (
                    <Badge variant="outline" className="text-[10px] px-1 py-0">
                      {msg.message_type}
                    </Badge>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-0.5 break-words">
                  {msg.content.length > 200
                    ? msg.content.slice(0, 200) + "..."
                    : msg.content}
                </p>
              </div>
            </motion.div>
          ))}

          {isLoading && (
            <div className="flex gap-2 items-center text-muted-foreground">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-xs">Agents working...</span>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="p-3 border-t border-border">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message the team..."
            className="text-sm"
          />
          <Button
            size="icon"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="h-9 w-9 flex-shrink-0"
          >
            <Send className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Trash2,
  Wrench,
  ChevronDown,
  ChevronUp,
  Loader2,
  Bot,
  User,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  type AgentInfo,
  type AgentChatResponse,
  AGENT_ROLES,
} from "@/lib/types";
import { chatWithAgent, clearAgentHistory } from "@/lib/api";
import { cn, generateId } from "@/lib/utils";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls?: AgentChatResponse["response"]["tool_calls"];
  sources?: AgentChatResponse["response"]["sources"];
  reasoning?: string[];
  iterations?: number;
  timestamp: number;
}

export default function AgentChat({ agent }: { agent: AgentInfo }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const roleConfig =
    AGENT_ROLES[agent.config.role] || AGENT_ROLES.custom;

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Clear messages when agent changes
  useEffect(() => {
    setMessages([]);
  }, [agent.id]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    setInput("");
    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const result = await chatWithAgent(agent.id, text);

      const assistantMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        content: result.response.content,
        toolCalls: result.response.tool_calls,
        sources: result.response.sources,
        reasoning: result.response.reasoning,
        iterations: result.response.iterations,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : "Failed to get response. Make sure the backend is running."}`,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMsg]);
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

  const handleClear = async () => {
    try {
      await clearAgentHistory(agent.id);
    } catch {
      // Ignore backend errors for clear
    }
    setMessages([]);
  };

  const toggleToolExpand = (msgId: string) => {
    setExpandedTools((prev) => {
      const next = new Set(prev);
      if (next.has(msgId)) next.delete(msgId);
      else next.add(msgId);
      return next;
    });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center text-xl"
            style={{ backgroundColor: `${roleConfig.color}20` }}
          >
            {roleConfig.icon}
          </div>
          <div>
            <h3 className="font-semibold">{agent.config.name}</h3>
            <p className="text-xs text-muted-foreground">
              {agent.config.description || roleConfig.label} ·{" "}
              {agent.config.tools.length} tools
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            {roleConfig.label}
          </Badge>
          <Button variant="ghost" size="sm" onClick={handleClear}>
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="space-y-4 max-w-3xl mx-auto">
          {messages.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center text-3xl mx-auto mb-4"
                style={{ backgroundColor: `${roleConfig.color}15` }}
              >
                {roleConfig.icon}
              </div>
              <p className="font-medium text-foreground">
                Chat with {agent.config.name}
              </p>
              <p className="text-sm mt-1 max-w-md mx-auto">
                {agent.config.description ||
                  "Ask a question to get started."}
              </p>
              <div className="flex flex-wrap gap-2 justify-center mt-4">
                {agent.config.tools.slice(0, 4).map((tool) => (
                  <Badge key={tool} variant="secondary" className="text-xs">
                    <Wrench className="w-3 h-3 mr-1" />
                    {tool}
                  </Badge>
                ))}
                {agent.config.tools.length > 4 && (
                  <Badge variant="secondary" className="text-xs">
                    +{agent.config.tools.length - 4} more
                  </Badge>
                )}
              </div>
            </div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn(
                  "flex gap-3",
                  msg.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                {msg.role === "assistant" && (
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center text-sm flex-shrink-0 mt-1"
                    style={{ backgroundColor: `${roleConfig.color}20` }}
                  >
                    {roleConfig.icon}
                  </div>
                )}

                <div
                  className={cn(
                    "max-w-[80%] rounded-xl p-3",
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  )}
                >
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>

                  {/* Tool Calls */}
                  {msg.toolCalls && msg.toolCalls.length > 0 && (
                    <div className="mt-2">
                      <button
                        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                        onClick={() => toggleToolExpand(msg.id)}
                      >
                        <Wrench className="w-3 h-3" />
                        {msg.toolCalls.length} tool call
                        {msg.toolCalls.length > 1 ? "s" : ""}
                        {expandedTools.has(msg.id) ? (
                          <ChevronUp className="w-3 h-3" />
                        ) : (
                          <ChevronDown className="w-3 h-3" />
                        )}
                      </button>
                      {expandedTools.has(msg.id) && (
                        <div className="mt-2 space-y-1">
                          {msg.toolCalls.map((tc, i) => (
                            <div
                              key={i}
                              className="text-xs bg-background/50 rounded p-2"
                            >
                              <span className="font-mono font-medium">
                                {tc.tool}
                              </span>
                              <span
                                className={cn(
                                  "ml-2",
                                  tc.success
                                    ? "text-green-500"
                                    : "text-red-500"
                                )}
                              >
                                {tc.success ? "✓" : "✗"}
                              </span>
                              <p className="text-muted-foreground mt-1 truncate">
                                {tc.result_preview}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {msg.sources.slice(0, 3).map((src, i) => (
                        <Badge
                          key={i}
                          variant="outline"
                          className="text-xs"
                        >
                          <FileText className="w-3 h-3 mr-1" />
                          {src.document_id || "source"}
                        </Badge>
                      ))}
                    </div>
                  )}

                  {/* Iterations */}
                  {msg.iterations && msg.iterations > 1 && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {msg.iterations} reasoning steps
                    </p>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0 mt-1">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex gap-3"
            >
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center text-sm flex-shrink-0"
                style={{ backgroundColor: `${roleConfig.color}20` }}
              >
                {roleConfig.icon}
              </div>
              <div className="bg-muted rounded-xl p-3 flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm text-muted-foreground">
                  Thinking...
                </span>
              </div>
            </motion.div>
          )}
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="p-4 border-t border-border">
        <div className="max-w-3xl mx-auto flex gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Message ${agent.config.name}...`}
            className="min-h-[44px] max-h-32 resize-none"
            rows={1}
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            size="icon"
            className="h-11 w-11 flex-shrink-0"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

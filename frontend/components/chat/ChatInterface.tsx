"use client";

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Sparkles,
  ChevronDown,
  FileText,
  AlertTriangle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { PERSONAS, type PersonaId, type Message } from "@/lib/types";
import { useChatStore } from "@/lib/store";
import { findSimilarProblems, listProblems } from "@/lib/api";
import { cn, generateId } from "@/lib/utils";

// Persona Selector Component
function PersonaSelector() {
  const { activePersonaId, setActivePersona } = useChatStore();
  const activePersona = PERSONAS[activePersonaId];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className="h-auto p-2 gap-2 hover:bg-muted/50"
        >
          <Avatar className="w-8 h-8">
            <AvatarFallback
              style={{ backgroundColor: `${activePersona.color}30` }}
              className="text-lg"
            >
              {activePersona.avatar}
            </AvatarFallback>
          </Avatar>
          <div className="text-left">
            <p className="text-sm font-medium">{activePersona.name}</p>
            <p className="text-xs text-muted-foreground">{activePersona.role}</p>
          </div>
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-64">
        {(Object.keys(PERSONAS) as PersonaId[]).map((id) => {
          const persona = PERSONAS[id];
          return (
            <DropdownMenuItem
              key={id}
              onClick={() => setActivePersona(id)}
              className={cn(
                "p-3 cursor-pointer",
                activePersonaId === id && "bg-muted"
              )}
            >
              <Avatar className="w-8 h-8 mr-3">
                <AvatarFallback
                  style={{ backgroundColor: `${persona.color}30` }}
                  className="text-lg"
                >
                  {persona.avatar}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1">
                <p className="font-medium text-sm">{persona.name}</p>
                <p className="text-xs text-muted-foreground">{persona.role}</p>
              </div>
              {activePersonaId === id && (
                <Badge variant="default" className="text-xs">Active</Badge>
              )}
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "text-red-400",
  high: "text-orange-400",
  medium: "text-yellow-400",
  low: "text-green-400",
};

// Message Bubble Component
function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const persona = message.personaId ? PERSONAS[message.personaId] : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("flex gap-3", isUser && "flex-row-reverse")}
    >
      {!isUser && persona && (
        <Avatar className="w-8 h-8 flex-shrink-0">
          <AvatarFallback
            style={{ backgroundColor: `${persona.color}30` }}
            className="text-sm"
          >
            {persona.avatar}
          </AvatarFallback>
        </Avatar>
      )}

      <div className={cn("flex-1 max-w-[80%]", isUser && "text-right")}>
        {!isUser && persona && (
          <div className="flex items-center gap-2 mb-1">
            <span
              className="text-sm font-medium"
              style={{ color: persona.color }}
            >
              {persona.name}
            </span>
            <span className="text-xs text-muted-foreground">
              {new Date(message.timestamp).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </div>
        )}

        <div
          className={cn(
            "rounded-2xl px-4 py-2 inline-block text-left",
            isUser
              ? "bg-primary text-primary-foreground rounded-br-md"
              : "bg-muted rounded-bl-md"
          )}
        >
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          {message.isStreaming && (
            <span className="inline-block w-2 h-4 bg-current animate-pulse ml-1" />
          )}
        </div>

        {/* Problem sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 space-y-2">
            <p className="text-xs text-muted-foreground">Related problems:</p>
            {message.sources.map((problem, idx) => (
              <div
                key={idx}
                className="text-left p-2 rounded-lg border border-border bg-card text-xs"
              >
                <div className="flex items-center gap-2 mb-1">
                  <AlertTriangle
                    className={cn(
                      "w-3 h-3",
                      SEVERITY_COLORS[problem.severity] || "text-muted-foreground"
                    )}
                  />
                  <span className="font-medium">{problem.problem_statement}</span>
                  <Badge variant="outline" className="text-xs ml-auto">
                    {problem.severity}
                  </Badge>
                </div>
                <p className="text-muted-foreground italic">
                  &ldquo;{problem.quote_text.slice(0, 120)}
                  {problem.quote_text.length > 120 ? "..." : ""}&rdquo;
                </p>
                {problem.tags.length > 0 && (
                  <div className="flex gap-1 mt-1">
                    {problem.tags.map((t) => (
                      <Badge key={t} variant="secondary" className="text-xs py-0">
                        {t}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// Main Chat Interface Component
export default function ChatInterface() {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const {
    messages,
    addMessage,
    activePersonaId,
    isStreaming,
    setStreaming,
  } = useChatStore();

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;
    const query = input.trim();

    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: query,
      timestamp: new Date().toISOString(),
    };
    addMessage(userMessage);
    setInput("");
    setStreaming(true);

    try {
      // Try similarity search first; fall back to keyword search
      let responseText = "";
      let sources: Message["sources"] = [];

      try {
        const similar = await findSimilarProblems(query, 5, 0.3);
        if (similar.results.length > 0) {
          sources = similar.results.map((r) => r.problem);
          responseText = `I found ${similar.results.length} related problem${similar.results.length > 1 ? "s" : ""} in your evidence:\n\n`;
          similar.results.forEach((r, i) => {
            responseText += `${i + 1}. **${r.problem.problem_statement}** (${r.problem.severity}, score: ${(r.score * 100).toFixed(0)}%)\n`;
          });
        }
      } catch {
        // Embeddings may not exist yet — fall back to listing
      }

      if (!sources || sources.length === 0) {
        // Fall back to listing problems
        const problems = await listProblems(1, 10);
        if (problems.items.length > 0) {
          sources = problems.items;
          responseText = `I couldn't do a semantic search (embeddings may not be generated yet), but here are the ${problems.total} problems extracted so far:\n\n`;
          problems.items.forEach((p, i) => {
            responseText += `${i + 1}. **${p.problem_statement}** (${p.severity})\n`;
          });
          responseText += `\nTo enable semantic search, run the "Embed Problems" job from the API.`;
        } else {
          responseText =
            "No problems have been extracted yet. Add some evidence first, then extract problems from it.";
        }
      }

      const aiMessage: Message = {
        id: generateId(),
        role: "assistant",
        content: responseText,
        personaId: activePersonaId,
        timestamp: new Date().toISOString(),
        sources,
      };
      addMessage(aiMessage);
    } catch (err) {
      const errorMessage: Message = {
        id: generateId(),
        role: "assistant",
        content: `Sorry, I encountered an error: ${err instanceof Error ? err.message : "Unknown error"}`,
        personaId: activePersonaId,
        timestamp: new Date().toISOString(),
      };
      addMessage(errorMessage);
    } finally {
      setStreaming(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const activePersona = PERSONAS[activePersonaId];

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <PersonaSelector />
        <Badge variant="outline" className="text-xs">
          <Sparkles className="w-3 h-3 mr-1" />
          Problem Search
        </Badge>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="space-y-6">
          {messages.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-12"
            >
              <div
                className="w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center text-3xl"
                style={{ backgroundColor: `${activePersona.color}20` }}
              >
                {activePersona.avatar}
              </div>
              <h3 className="text-lg font-semibold mb-2">
                Search Problems
              </h3>
              <p className="text-muted-foreground text-sm max-w-md mx-auto mb-4">
                Ask about user problems extracted from your evidence. I&apos;ll find the most relevant ones.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "What are the most critical issues?",
                  "Show performance problems",
                  "What do users struggle with?",
                ].map((suggestion) => (
                  <Button
                    key={suggestion}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    onClick={() => setInput(suggestion)}
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>
            </motion.div>
          )}

          <AnimatePresence>
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </AnimatePresence>

          {isStreaming && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-2 text-muted-foreground"
            >
              <Avatar className="w-6 h-6">
                <AvatarFallback
                  style={{ backgroundColor: `${activePersona.color}30` }}
                  className="text-xs"
                >
                  {activePersona.avatar}
                </AvatarFallback>
              </Avatar>
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                <span
                  className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                  style={{ animationDelay: "0.1s" }}
                />
                <span
                  className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                  style={{ animationDelay: "0.2s" }}
                />
              </div>
            </motion.div>
          )}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="p-4 border-t border-border">
        <div className="flex items-end gap-2">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search for problems..."
              className="min-h-[44px] max-h-[200px] pr-12 resize-none"
              rows={1}
            />
            <Button
              size="icon"
              className="absolute right-2 bottom-2 h-8 w-8"
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-2 text-center">
          Press Enter to search, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

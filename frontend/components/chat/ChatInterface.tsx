"use client";

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Paperclip,
  Sparkles,
  ChevronDown,
  FileText,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Upload,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { PERSONAS, type PersonaId, type Message, type HumanTask } from "@/lib/types";
import { useChatStore } from "@/lib/store";
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
                <Badge variant="success" className="text-xs">Active</Badge>
              )}
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

// Human Task Card Component
function HumanTaskCard({ task }: { task: HumanTask }) {
  const statusColors = {
    pending: "text-yellow-400 bg-yellow-400/10",
    "in-progress": "text-blue-400 bg-blue-400/10",
    completed: "text-green-400 bg-green-400/10",
    cancelled: "text-red-400 bg-red-400/10",
  };

  const StatusIcon = {
    pending: Clock,
    "in-progress": Sparkles,
    completed: CheckCircle2,
    cancelled: AlertTriangle,
  }[task.status];

  return (
    <Card className="border-yellow-500/30 bg-yellow-500/5">
      <CardContent className="p-4">
        <div className="flex items-start gap-3 mb-3">
          <div className={cn("p-2 rounded-lg", statusColors[task.status])}>
            <StatusIcon className="w-4 h-4" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h4 className="font-semibold text-sm">{task.title}</h4>
              <Badge
                variant={task.status === "completed" ? "success" : "warning"}
                className="text-xs"
              >
                {task.status}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">{task.type}</p>
          </div>
        </div>

        {/* Instructions */}
        <div className="space-y-2 mb-4">
          <p className="text-xs font-medium text-muted-foreground">
            Instructions:
          </p>
          <ol className="space-y-1">
            {task.instructions.map((instruction, idx) => (
              <li key={idx} className="flex gap-2 text-sm">
                <span className="text-muted-foreground">{idx + 1}.</span>
                <span>{instruction}</span>
              </li>
            ))}
          </ol>
        </div>

        {/* Safety Warnings */}
        {task.safetyWarnings && task.safetyWarnings.length > 0 && (
          <div className="space-y-2 mb-4">
            <p className="text-xs font-medium text-yellow-400 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" />
              Safety Warnings:
            </p>
            <ul className="space-y-1">
              {task.safetyWarnings.map((warning, idx) => (
                <li
                  key={idx}
                  className="text-sm text-yellow-200/80 flex items-start gap-2"
                >
                  <span>•</span>
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Expected Output */}
        <p className="text-xs text-muted-foreground mb-4">
          <span className="font-medium">Expected output:</span>{" "}
          {task.expectedOutput}
        </p>

        {/* Upload/Complete Button */}
        {task.status !== "completed" && (
          <Button className="w-full" size="sm">
            <Upload className="w-4 h-4 mr-2" />
            Upload Results & Complete Task
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

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
      {/* Avatar */}
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
        {/* Name & Time */}
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

        {/* Message Content */}
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

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-xs text-muted-foreground">Sources:</p>
            <div className="flex flex-wrap gap-1">
              {message.sources.map((source, idx) => (
                <Badge
                  key={idx}
                  variant="outline"
                  className="text-xs cursor-pointer hover:bg-muted"
                >
                  <FileText className="w-3 h-3 mr-1" />
                  {source.documentName}
                  <Badge variant="info" className="ml-1 text-xs py-0 px-1">
                    L{source.layer}
                  </Badge>
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Human Task */}
        {message.humanTask && (
          <div className="mt-3">
            <HumanTaskCard task={message.humanTask} />
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
  const { messages, addMessage, activePersonaId, isStreaming } = useChatStore();

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Handle send message
  const handleSend = () => {
    if (!input.trim() || isStreaming) return;

    // Add user message
    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: input.trim(),
      timestamp: new Date().toISOString(),
    };
    addMessage(userMessage);

    // Simulate AI response (in production, this would call the API)
    setTimeout(() => {
      const persona = PERSONAS[activePersonaId];
      const aiMessage: Message = {
        id: generateId(),
        role: "assistant",
        content: `${persona.greeting}\n\nI've analyzed your question about "${input.slice(0, 50)}..."\n\nBased on the documents in your knowledge base, here's what I found:\n\n1. The RAPTOR tree structure allows for both detailed and high-level retrieval\n2. Layer 0 contains the original document chunks\n3. Higher layers contain summarized information\n\nWould you like me to dive deeper into any specific aspect?`,
        personaId: activePersonaId,
        timestamp: new Date().toISOString(),
        sources: [
          {
            documentId: "doc-1",
            documentName: "RAPTOR_Paper.pdf",
            chunkId: "chunk-1",
            content: "RAPTOR builds a hierarchical tree structure...",
            layer: 0,
            relevanceScore: 0.92,
          },
          {
            documentId: "doc-2",
            documentName: "Implementation_Notes.md",
            chunkId: "chunk-2",
            content: "The collapsed tree method performs better...",
            layer: 1,
            relevanceScore: 0.87,
          },
        ],
      };
      addMessage(aiMessage);
    }, 1000);

    setInput("");
  };

  // Handle keyboard shortcuts
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
          RAPTOR Active
        </Badge>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="space-y-6">
          {/* Welcome message */}
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
                Chat with {activePersona.name}
              </h3>
              <p className="text-muted-foreground text-sm max-w-md mx-auto mb-4">
                {activePersona.description}
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Summarize all documents",
                  "Explain the main concepts",
                  "Find related topics",
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

          {/* Message list */}
          <AnimatePresence>
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </AnimatePresence>

          {/* Typing indicator */}
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
          <Button variant="ghost" size="icon" className="flex-shrink-0">
            <Paperclip className="w-4 h-4" />
          </Button>
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Ask ${activePersona.name} anything...`}
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
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

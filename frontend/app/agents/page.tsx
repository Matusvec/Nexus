"use client";

import { useState, useEffect } from "react";
import AgentsPanel from "@/components/agents/AgentsPanel";
import Sidebar from "@/components/layout/Sidebar";
import UploadModal from "@/components/documents/UploadModal";
import SearchCommand from "@/components/layout/SearchCommand";
import { useAgentsStore } from "@/lib/store";
import { getAgents } from "@/lib/api";
import type { AgentInfo, AgentRoleType } from "@/lib/types";

// Default agents to show when backend is not connected
const DEFAULT_AGENTS: AgentInfo[] = [
  {
    id: "research",
    config: {
      name: "Research Agent",
      role: "research" as AgentRoleType,
      system_prompt: "",
      description: "Searches the RAG knowledge base, synthesizes findings, and provides cited answers",
      tools: ["rag_search", "rag_tree_search", "document_list", "document_summary", "text_summarize", "extract_entities"],
      model: "",
      temperature: 0.7,
      max_iterations: 10,
    },
    created_at: Date.now() / 1000,
    message_count: 0,
    is_custom: false,
  },
  {
    id: "web_search",
    config: {
      name: "Web Search Agent",
      role: "web_search" as AgentRoleType,
      system_prompt: "",
      description: "Searches the web and YouTube for current information, articles, and videos",
      tools: ["web_search", "youtube_search", "rag_search", "text_summarize"],
      model: "",
      temperature: 0.7,
      max_iterations: 10,
    },
    created_at: Date.now() / 1000,
    message_count: 0,
    is_custom: false,
  },
  {
    id: "code",
    config: {
      name: "Code Agent",
      role: "code" as AgentRoleType,
      system_prompt: "",
      description: "Assists with code analysis, debugging, and technical implementations",
      tools: ["rag_search", "calculate", "web_search", "text_summarize"],
      model: "",
      temperature: 0.7,
      max_iterations: 10,
    },
    created_at: Date.now() / 1000,
    message_count: 0,
    is_custom: false,
  },
  {
    id: "document",
    config: {
      name: "Document Agent",
      role: "document" as AgentRoleType,
      system_prompt: "",
      description: "Analyzes documents in-depth, navigates the RAG hierarchy for detailed and summary information",
      tools: ["rag_search", "rag_tree_search", "document_list", "document_summary", "extract_entities", "text_summarize"],
      model: "",
      temperature: 0.7,
      max_iterations: 10,
    },
    created_at: Date.now() / 1000,
    message_count: 0,
    is_custom: false,
  },
];

export default function AgentsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { setAgents, agents } = useAgentsStore();

  useEffect(() => {
    // Try to load agents from backend
    getAgents()
      .then((data) => {
        setAgents(data);
      })
      .catch(() => {
        // Use defaults when backend is not available
        if (agents.length === 0) {
          setAgents(DEFAULT_AGENTS);
        }
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background">
      {/* Sidebar */}
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <AgentsPanel />
      </main>

      {/* Modals */}
      <UploadModal />
      <SearchCommand />
    </div>
  );
}

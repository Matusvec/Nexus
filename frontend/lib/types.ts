// ============================================
// NEXUS TYPE DEFINITIONS
// ============================================

// Document & Chunks
export interface Document {
  id: string;
  filename: string;
  uploadedAt: string;
  fileSize: number;
  chunkCount: number;
  status: "processing" | "ready" | "error";
  summary?: string;
  groupId?: string;
}

export interface Chunk {
  id: string;
  documentId: string;
  content: string;
  layer: number;
  contentType: "text" | "image" | "table";
  metadata: {
    pageNumber?: number;
    imageRefs?: string[];
    parentIds?: string[];
    childIds?: string[];
  };
}

// Canvas & Groups
export interface DocumentGroup {
  id: string;
  name: string;
  description?: string;
  color: string;
  documentIds: string[];
  position: { x: number; y: number };
  parentGroupId?: string;
  assignedPersona?: PersonaId;
}

export interface GroupConnection {
  id: string;
  sourceGroupId: string;
  targetGroupId: string;
  label?: string;
}

// AI Personas
export type PersonaId = "max" | "elena" | "byte" | "stacy";

export interface Persona {
  id: PersonaId;
  name: string;
  role: string;
  description: string;
  color: string;
  avatar: string;
  traits: string[];
  greeting: string;
}

export const PERSONAS: Record<PersonaId, Persona> = {
  max: {
    id: "max",
    name: "Max",
    role: "Mechanical Engineer",
    description: "Gruff, practical, safety-focused. Your go-to for CAD, materials, and manufacturing.",
    color: "#F97316",
    avatar: "🔧",
    traits: ["Practical", "Safety-focused", "Direct"],
    greeting: "Hey! Max here. What are we building today?",
  },
  elena: {
    id: "elena",
    name: "Dr. Elena",
    role: "Physicist",
    description: "Precise, encouraging, explains deeply. Expert in physics, math, and scientific analysis.",
    color: "#8B5CF6",
    avatar: "⚛️",
    traits: ["Precise", "Encouraging", "Deep thinker"],
    greeting: "Hello! Dr. Elena here. What fascinating problem shall we explore?",
  },
  byte: {
    id: "byte",
    name: "Byte",
    role: "Software Engineer",
    description: "Fast-talking, meme-savvy. Expert in code, algorithms, and system design.",
    color: "#10B981",
    avatar: "💻",
    traits: ["Fast-paced", "Witty", "Tech-savvy"],
    greeting: "Yo! Byte here. Ready to ship some code?",
  },
  stacy: {
    id: "stacy",
    name: "Stacy",
    role: "Electrical Engineer",
    description: "Methodical, diagram-obsessed. Expert in circuits, signals, and electronics.",
    color: "#3B82F6",
    avatar: "⚡",
    traits: ["Methodical", "Detail-oriented", "Systematic"],
    greeting: "Hi there! Stacy here. Let's trace through the circuit.",
  },
};

// Chat & Messages
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  personaId?: PersonaId;
  timestamp: string;
  sources?: MessageSource[];
  isStreaming?: boolean;
  humanTask?: HumanTask;
}

export interface MessageSource {
  documentId: string;
  documentName: string;
  chunkId: string;
  content: string;
  layer: number;
  relevanceScore: number;
}

// Human-in-the-Loop Tasks
export interface HumanTask {
  id: string;
  type: "measurement" | "photo" | "3d_print" | "solder" | "test" | "other";
  title: string;
  instructions: string[];
  safetyWarnings?: string[];
  expectedOutput: string;
  status: "pending" | "in-progress" | "completed" | "cancelled";
  uploadedFiles?: UploadedFile[];
  createdAt: string;
  completedAt?: string;
}

export interface UploadedFile {
  id: string;
  name: string;
  type: string;
  size: number;
  url: string;
  uploadedAt: string;
}

// Query
export interface QueryResult {
  answer: string;
  sources: MessageSource[];
  queryType: "simple" | "complex" | "exploratory";
  tokensUsed: number;
}

// Statistics
export interface DatabaseStats {
  totalDocuments: number;
  totalChunks: number;
  chunksByLayer: Record<number, number>;
  documents: {
    id: string;
    name: string;
    chunkCount: number;
  }[];
}

// Upload Progress
export interface UploadProgress {
  stage: "parsing" | "chunking" | "embedding" | "tree-building" | "complete" | "error";
  progress: number;
  message: string;
  error?: string;
}

// Canvas Node Types for React Flow
export interface CanvasNode {
  id: string;
  type: "documentGroup";
  position: { x: number; y: number };
  data: DocumentGroup;
}

export interface CanvasEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  animated?: boolean;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// ============================================
// AGENTIC AI TYPES
// ============================================

// Agent Roles
export type AgentRoleType =
  | "research"
  | "code"
  | "web_search"
  | "document"
  | "custom"
  | "orchestrator";

// Agent Tool
export interface AgentTool {
  name: string;
  description: string;
  parameters: {
    name: string;
    type: string;
    description: string;
    required: boolean;
  }[];
  category: string;
}

// Agent Configuration
export interface AgentConfig {
  name: string;
  role: AgentRoleType;
  system_prompt: string;
  description: string;
  tools: string[];
  model: string;
  temperature: number;
  max_iterations: number;
}

// Agent
export interface AgentInfo {
  id: string;
  config: AgentConfig;
  created_at: number;
  message_count: number;
  is_custom?: boolean;
}

// Agent Chat Response
export interface AgentChatResponse {
  agent_id: string;
  agent_name: string;
  response: {
    content: string;
    tool_calls: {
      tool: string;
      args: Record<string, unknown>;
      result_preview: string;
      success: boolean;
    }[];
    sources: {
      chunk_id?: string;
      document_id?: string;
      layer?: number;
      score?: number;
      preview?: string;
    }[];
    reasoning: string[];
    agent_id: string;
    iterations: number;
  };
}

// Orchestrator Session Message
export interface OrchestratorMessage {
  id: string;
  sender: string;
  sender_name: string;
  content: string;
  timestamp: number;
  message_type: "message" | "tool_result" | "delegation" | "summary";
  metadata: Record<string, unknown>;
}

// Orchestrator Session
export interface OrchestratorSession {
  id: string;
  name: string;
  messages: OrchestratorMessage[];
  participating_agents: string[];
  created_at: number;
}

// Orchestrator Chat Response
export interface OrchestratorChatResponse {
  session_id: string;
  response: {
    content: string;
    tool_calls: unknown[];
    sources: unknown[];
    reasoning: string[];
    agent_id: string;
    iterations: number;
  };
  responding_agent: {
    id: string;
    name: string;
    role: string;
  };
  participating_agents?: {
    id: string;
    name: string;
    role: string;
  }[];
  session_messages: OrchestratorMessage[];
}

// Agent Role Display Config
export const AGENT_ROLES: Record<
  AgentRoleType,
  { label: string; icon: string; color: string }
> = {
  research: { label: "Research", icon: "🔍", color: "#8B5CF6" },
  code: { label: "Code", icon: "💻", color: "#10B981" },
  web_search: { label: "Web Search", icon: "🌐", color: "#3B82F6" },
  document: { label: "Document", icon: "📄", color: "#F59E0B" },
  custom: { label: "Custom", icon: "⚙️", color: "#EC4899" },
  orchestrator: { label: "Orchestrator", icon: "🎯", color: "#6366F1" },
};

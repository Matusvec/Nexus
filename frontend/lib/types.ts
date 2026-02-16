// ============================================
// NEXUS TYPE DEFINITIONS
// ============================================

// ---- Evidence (replaces Documents) ----
export type SourceType = "interview" | "support_ticket" | "sales_note" | "survey" | "other";

export interface Evidence {
  id: string;
  title: string;
  source_type: SourceType;
  persona: string | null;
  segment: string | null;
  source_date: string | null;
  chunk_count: number;
  created_at: string | null;
}

export interface EvidenceDetail extends Evidence {
  raw_text: string;
  chunks: EvidenceChunk[];
}

export interface EvidenceChunk {
  id: string;
  chunk_index: number;
  chunk_text: string;
  start_offset: number;
  end_offset: number;
  token_count: number | null;
}

export interface EvidenceListResponse {
  items: Evidence[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

// ---- Problems ----
export type Severity = "critical" | "high" | "medium" | "low";

export interface ProblemMention {
  id: string;
  evidence_id: string;
  chunk_id: string;
  problem_statement: string;
  severity: Severity;
  quote_text: string;
  quote_start: number | null;
  quote_end: number | null;
  persona: string | null;
  segment: string | null;
  tags: string[];
  created_at: string | null;
}

export interface ProblemMentionListResponse {
  items: ProblemMention[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface SimilarProblemResult {
  problem: ProblemMention;
  score: number;
}

export interface SimilarProblemsResponse {
  query_text: string;
  results: SimilarProblemResult[];
}

// ---- Jobs ----
export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface JobResponse {
  job_id: string;
  status: JobStatus;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  job_type: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  result_count: number | null;
}

// ---- Clusters ----
export interface ProblemCluster {
  id: string;
  label: string;
  summary: string | null;
  threshold: number;
  mention_count: number;
  tags: string[];
  created_at: string | null;
  updated_at: string | null;
}

export interface ClusterMember {
  id: string;
  problem_id: string;
  similarity: number;
}

export interface ClusterDetail extends ProblemCluster {
  members: ClusterMember[];
  proposals: FeatureProposal[];
}

export interface FeatureProposal {
  id: string;
  cluster_id: string;
  title: string;
  description: string | null;
  priority_score: number | null;
  impact: string | null;
  effort: string | null;
  version: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface RoadmapItem {
  proposal: FeatureProposal;
  cluster_label: string;
  mention_count: number;
  priority_score: number | null;
}

export interface RoadmapResponse {
  items: RoadmapItem[];
  total: number;
}

// ---- Problem Stats ----
export interface ProblemStats {
  total: number;
  by_severity: Record<string, number>;
  by_persona: Record<string, number>;
  by_tag: Record<string, number>;
}

// ---- Legacy types kept for canvas/UI ----

// Canvas & Groups (now backed by clusters)
export interface DocumentGroup {
  id: string;
  name: string;
  description?: string;
  color: string;
  documentIds: string[];
  position: { x: number; y: number };
  parentGroupId?: string;
  assignedPersona?: PersonaId;
  [key: string]: unknown;
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
  sources?: ProblemMention[];
  isStreaming?: boolean;
}

// Query
export interface QueryResult {
  answer: string;
  sources: ProblemMention[];
}

// Upload Progress
export interface UploadProgress {
  stage: "submitting" | "extracting" | "complete" | "error";
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

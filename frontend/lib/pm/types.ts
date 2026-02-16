// ============================================
// PM PIPELINE TYPE DEFINITIONS
// ============================================

export type SourceType = "interview" | "support_ticket" | "sales_note" | "survey" | "other";
export type Severity = "critical" | "high" | "medium" | "low";
export type JobStatus = "pending" | "running" | "completed" | "failed";

// ── Evidence ──

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

// ── Problems ──

export interface ProblemMention {
  id: string;
  evidence_id: string;
  chunk_id: string | null;
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

export interface SimilarProblem {
  problem: ProblemMention;
  score: number;
}

export interface ProblemStats {
  total: number;
  by_severity: Record<string, number>;
  by_persona: Record<string, number>;
  by_tag: Record<string, number>;
  by_source_type: Record<string, number>;
}

// ── Clusters ──

export interface Cluster {
  id: string;
  label: string;
  summary: string | null;
  threshold: number;
  tags: string[];
  mention_count: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface ClusterMember {
  id: string;
  problem_id: string;
  similarity: number;
}

export interface ClusterDetail extends Cluster {
  members: ClusterMember[];
  proposals: Proposal[];
}

// ── Proposals ──

export interface Proposal {
  id: string;
  cluster_id: string;
  title: string;
  description: string;
  priority_score: number | null;
  impact: string | null;
  effort: string | null;
  version: number;
  created_at: string | null;
  updated_at: string | null;
}

// ── Roadmap ──

export interface RoadmapItem {
  proposal: Proposal;
  cluster_label: string;
  mention_count: number;
  priority_score: number | null;
}

export interface RoadmapResponse {
  items: RoadmapItem[];
  total: number;
}

// ── Jobs ──

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

// ── Paginated ──

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

// ── Pipeline ──

export interface PipelineStep {
  label: string;
  count: number | null;
  status: "complete" | "running" | "pending" | "error";
}


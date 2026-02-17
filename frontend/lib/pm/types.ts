// ============================================
// PM PIPELINE TYPE DEFINITIONS — Strategy §8
// ============================================

export type SourceType = "interview" | "support_ticket" | "sales_note" | "survey" | "other";
export type Severity = "critical" | "high" | "medium" | "low";
export type JobStatus = "pending" | "running" | "completed" | "failed";
export type ProposalStatus = "draft" | "approved" | "rejected" | "archived";
export type ScopeEstimate = "S" | "M" | "L" | "XL";
export type TaskCategory = "backend" | "frontend" | "data" | "qa";
export type TaskEffort = "XS" | "S" | "M" | "L" | "XL";

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
  severity_distribution?: Record<string, number>;
  top_quote?: string;
  top_quote_source?: string;
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
  status?: ProposalStatus;
  created_at: string | null;
  updated_at: string | null;
}

export interface ProposalDetail extends Proposal {
  user_story: string | null;
  jtbd_framing: string | null;
  rationale: string;
  success_metrics: SuccessMetric[];
  risks: Risk[];
  edge_cases: string[];
  scope_estimate: ScopeEstimate;
  citations: Citation[];
  cluster: Cluster;
  tasks_generated: boolean;
}

export interface SuccessMetric {
  metric: string;
  target: string;
  reasoning: string;
}

export interface Risk {
  risk: string;
  mitigation: string;
  severity: "high" | "medium" | "low";
}

export interface Citation {
  id: string;
  problem_id: string;
  citation_context: string;
  quote_text: string;
  evidence_title: string;
}

// ── Tasks ──

export interface Task {
  id: string;
  proposal_id: string;
  parent_task_id: string | null;
  title: string;
  description: string | null;
  category: TaskCategory;
  acceptance_criteria: string[];
  estimated_effort: TaskEffort | null;
  dependencies: string[];
  sort_order: number;
  subtasks: Task[];
}

export interface TaskTree {
  proposal_id: string;
  feature_name: string;
  backend: Task[];
  frontend: Task[];
  data: Task[];
  qa: Task[];
  total_tasks: number;
}

// ── Roadmap ──

export interface RoadmapItem {
  proposal: Proposal;
  cluster_label: string;
  mention_count: number;
  priority_score: number | null;
  frequency_score?: number;
  severity_score?: number;
  strategic_weight?: number;
  effort_estimate?: number;
}

export interface RoadmapResponse {
  items: RoadmapItem[];
  total: number;
}

export interface ScoreBreakdown {
  formula: string;
  frequency: { value: number; explanation: string };
  severity: { value: number; distribution: Record<Severity, number> };
  weight: { value: number; reason: string };
  effort: { value: number; scope: ScopeEstimate };
  final: number;
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

// ── LLM Costs ──

export interface CostResponse {
  total_calls: number;
  total_cost_usd: number;
  total_input_tokens: number;
  total_output_tokens: number;
  by_model: Record<
    string,
    {
      calls: number;
      input_tokens: number;
      output_tokens: number;
      cost_usd: number;
    }
  >;
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


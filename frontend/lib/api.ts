// ============================================
// NEXUS API CLIENT
// ============================================
// Calls the FastAPI backend via Next.js rewrite proxy (/api/* → backend).

import type {
  Evidence,
  EvidenceDetail,
  EvidenceListResponse,
  SourceType,
  ProblemMention,
  ProblemMentionListResponse,
  SimilarProblemsResponse,
  ProblemStats,
  JobResponse,
  JobStatusResponse,
  ProblemCluster,
  ClusterDetail,
  FeatureProposal,
  RoadmapResponse,
} from "./types";

const API_PREFIX = "/api/v1";

// Helper function for API calls
async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_PREFIX}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `API Error: ${response.status}`);
  }

  return response.json();
}

// ============================================
// EVIDENCE ENDPOINTS
// ============================================

export async function createEvidence(data: {
  title: string;
  source_type: SourceType;
  raw_text: string;
  persona?: string;
  segment?: string;
  source_date?: string;
  metadata?: Record<string, unknown>;
}): Promise<Evidence> {
  return fetchApi<Evidence>("/evidence", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function listEvidence(
  page = 1,
  perPage = 20,
  filters?: { source_type?: SourceType; persona?: string; segment?: string }
): Promise<EvidenceListResponse> {
  const params = new URLSearchParams({
    page: String(page),
    per_page: String(perPage),
  });
  if (filters?.source_type) params.set("source_type", filters.source_type);
  if (filters?.persona) params.set("persona", filters.persona);
  if (filters?.segment) params.set("segment", filters.segment);
  return fetchApi<EvidenceListResponse>(`/evidence?${params}`);
}

export async function getEvidence(id: string): Promise<EvidenceDetail> {
  return fetchApi<EvidenceDetail>(`/evidence/${id}`);
}

export async function deleteEvidence(id: string): Promise<void> {
  await fetchApi(`/evidence/${id}`, { method: "DELETE" });
}

// ============================================
// PROBLEMS ENDPOINTS
// ============================================

export async function listProblems(
  page = 1,
  perPage = 20,
  filters?: { severity?: string; persona?: string; tag?: string; evidence_id?: string }
): Promise<ProblemMentionListResponse> {
  const params = new URLSearchParams({
    page: String(page),
    per_page: String(perPage),
  });
  if (filters?.severity) params.set("severity", filters.severity);
  if (filters?.persona) params.set("persona", filters.persona);
  if (filters?.tag) params.set("tag", filters.tag);
  if (filters?.evidence_id) params.set("evidence_id", filters.evidence_id);
  return fetchApi<ProblemMentionListResponse>(`/problems?${params}`);
}

export async function getProblem(id: string): Promise<ProblemMention> {
  return fetchApi<ProblemMention>(`/problems/${id}`);
}

export async function findSimilarProblems(
  text: string,
  limit = 10,
  minScore = 0.5
): Promise<SimilarProblemsResponse> {
  const params = new URLSearchParams({
    text,
    limit: String(limit),
    min_score: String(minScore),
  });
  return fetchApi<SimilarProblemsResponse>(`/problems/similar?${params}`);
}

export async function getProblemStats(filters?: {
  persona?: string;
  severity?: string;
  tag?: string;
}): Promise<ProblemStats> {
  const params = new URLSearchParams();
  if (filters?.persona) params.set("persona", filters.persona);
  if (filters?.severity) params.set("severity", filters.severity);
  if (filters?.tag) params.set("tag", filters.tag);
  const qs = params.toString();
  return fetchApi<ProblemStats>(`/problems/stats${qs ? `?${qs}` : ""}`);
}

// ============================================
// JOBS ENDPOINTS
// ============================================

export async function extractProblems(evidenceId: string): Promise<JobResponse> {
  return fetchApi<JobResponse>("/jobs/extract_problems", {
    method: "POST",
    body: JSON.stringify({ evidence_id: evidenceId }),
  });
}

export async function embedProblems(problemIds?: string[]): Promise<JobResponse> {
  return fetchApi<JobResponse>("/jobs/embed_problems", {
    method: "POST",
    body: JSON.stringify(problemIds ? { problem_ids: problemIds } : {}),
  });
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return fetchApi<JobStatusResponse>(`/jobs/${jobId}/status`);
}

export async function getLLMCosts(): Promise<Record<string, unknown>> {
  return fetchApi("/llm/costs");
}

// ============================================
// CLUSTERS ENDPOINTS
// ============================================

export async function runClustering(threshold = 0.75): Promise<{ clusters_created: number }> {
  return fetchApi("/clusters/run", {
    method: "POST",
    body: JSON.stringify({ threshold }),
  });
}

export async function listClusters(
  page = 1,
  perPage = 20
): Promise<{ items: ProblemCluster[]; total: number; page: number; per_page: number; total_pages: number }> {
  const params = new URLSearchParams({
    page: String(page),
    per_page: String(perPage),
  });
  return fetchApi(`/clusters?${params}`);
}

export async function getCluster(id: string): Promise<ClusterDetail> {
  return fetchApi<ClusterDetail>(`/clusters/${id}`);
}

export async function createProposal(data: {
  cluster_id: string;
  title: string;
  description?: string;
  priority_score?: number;
  impact?: string;
  effort?: string;
}): Promise<FeatureProposal> {
  return fetchApi<FeatureProposal>("/proposals", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getRoadmap(limit = 50): Promise<RoadmapResponse> {
  return fetchApi<RoadmapResponse>(`/roadmap?limit=${limit}`);
}

// ============================================
// HEALTH
// ============================================

export async function healthCheck(): Promise<{ status: string }> {
  return fetchApi<{ status: string }>("/health");
}


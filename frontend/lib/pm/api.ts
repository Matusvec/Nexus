// ============================================
// PM PIPELINE API CLIENT
// ============================================
//
// Usage from SERVER components : pmFetch("/evidence?page=1")
// Usage from CLIENT components : pmFetch("/evidence?page=1")  (same!)
//
// On the server the Next.js rewrite doesn't exist, so we use
// the absolute backend URL.  In the browser the rewrite proxies
// /api/v1/* to the backend automatically.

import type {
  Evidence,
  EvidenceDetail,
  ProblemMention,
  ProblemStats,
  SimilarProblem,
  Cluster,
  ClusterDetail,
  Proposal,
  RoadmapResponse,
  JobResponse,
  JobStatusResponse,
  PaginatedResponse,
} from "./types";

const BACKEND_ORIGIN =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const PREFIX = "/api/v1";

function baseUrl(): string {
  // Client (browser) → use relative URL so the Next.js rewrite kicks in
  if (typeof window !== "undefined") return PREFIX;
  // Server (SSR / RSC) → need absolute URL
  return `${BACKEND_ORIGIN}${PREFIX}`;
}

export async function pmFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = `${baseUrl()}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  if (res.status === 204) return undefined as unknown as T;
  return res.json() as Promise<T>;
}

export async function pmFetchSafe<T>(
  path: string,
  init?: RequestInit,
): Promise<T | null> {
  try {
    return await pmFetch<T>(path, { ...init, cache: "no-store" });
  } catch {
    return null;
  }
}

// ── Evidence ──

export const getEvidence = (page = 1, perPage = 20) =>
  pmFetch<PaginatedResponse<Evidence>>(
    `/evidence?page=${page}&per_page=${perPage}`,
  );

export const getEvidenceDetail = (id: string) =>
  pmFetch<EvidenceDetail>(`/evidence/${id}`);

export const createEvidence = (data: {
  title: string;
  source_type: string;
  persona?: string;
  segment?: string;
  source_date?: string;
  raw_text: string;
}) =>
  pmFetch<Evidence>("/evidence", {
    method: "POST",
    body: JSON.stringify(data),
  });

export const deleteEvidence = (id: string) =>
  pmFetch<void>(`/evidence/${id}`, { method: "DELETE" });

// ── Problems ──

export const getProblems = (
  params: Record<string, string | number | undefined> = {},
) => {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== "") qs.set(k, String(v));
  });
  if (!qs.has("page")) qs.set("page", "1");
  if (!qs.has("per_page")) qs.set("per_page", "20");
  return pmFetch<PaginatedResponse<ProblemMention>>(
    `/problems?${qs.toString()}`,
  );
};

export const getProblem = (id: string) =>
  pmFetch<ProblemMention>(`/problems/${id}`);

export const getProblemStats = () =>
  pmFetch<ProblemStats>("/problems/stats");

export const getSimilarProblems = (text: string, limit = 10) =>
  pmFetch<{ query_text: string; results: SimilarProblem[] }>(
    `/problems/similar?text=${encodeURIComponent(text)}&limit=${limit}`,
  );

// ── Jobs ──

export const extractProblems = (evidenceId: string) =>
  pmFetch<JobResponse>("/jobs/extract_problems", {
    method: "POST",
    body: JSON.stringify({ evidence_id: evidenceId }),
  });

export const embedProblems = (limit?: number) =>
  pmFetch<JobResponse>("/jobs/embed_problems", {
    method: "POST",
    body: JSON.stringify(limit ? { limit } : {}),
  });

export const getJobStatus = (jobId: string) =>
  pmFetch<JobStatusResponse>(`/jobs/${jobId}/status`);

export const getLLMCosts = () =>
  pmFetch<Record<string, unknown>>("/llm/costs");

export const getLLMCalls = () =>
  pmFetch<Record<string, unknown>[]>("/llm/calls");

// ── Clusters ──

export const runClustering = (threshold = 0.75) =>
  pmFetch<{ clusters_created: number }>(`/clusters/run?threshold=${threshold}`, {
    method: "POST",
  });

export const getClusters = (page = 1, perPage = 20) =>
  pmFetch<PaginatedResponse<Cluster>>(
    `/clusters?page=${page}&per_page=${perPage}`,
  );

export const getClusterDetail = (id: string) =>
  pmFetch<ClusterDetail>(`/clusters/${id}`);

// ── Proposals ──

export const createProposal = (data: {
  cluster_id: string;
  title: string;
  description: string;
  priority_score?: number;
  impact?: string;
  effort?: string;
}) =>
  pmFetch<Proposal>("/proposals", {
    method: "POST",
    body: JSON.stringify(data),
  });

// ── Roadmap ──

export const getRoadmap = () => pmFetch<RoadmapResponse>("/roadmap");

// ── Health ──

export const healthCheck = () =>
  pmFetch<{ status: string }>("/health");


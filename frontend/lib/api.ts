// ============================================
// NEXUS API CLIENT
// ============================================
// This file contains all API calls to the FastAPI backend.
// Replace BASE_URL with your actual backend URL.

import type {
  Document,
  DatabaseStats,
  QueryResult,
  UploadProgress,
  HumanTask,
  DocumentGroup,
  Chunk,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Helper function for API calls
async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `API Error: ${response.status}`);
  }

  return response.json();
}

// ============================================
// DOCUMENT ENDPOINTS
// ============================================

export async function uploadDocument(
  file: File,
  groupId?: string,
  onProgress?: (progress: UploadProgress) => void
): Promise<Document> {
  const formData = new FormData();
  formData.append("file", file);
  if (groupId) formData.append("group_id", groupId);

  const response = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Upload failed");
  }

  // For now, return the response directly
  // In production, you might use Server-Sent Events for progress
  return response.json();
}

export async function getDocuments(): Promise<Document[]> {
  return fetchApi<Document[]>("/documents");
}

export async function getDocument(id: string): Promise<Document> {
  return fetchApi<Document>(`/documents/${id}`);
}

export async function deleteDocument(id: string): Promise<void> {
  await fetchApi(`/documents/${id}`, { method: "DELETE" });
}

export async function getDocumentChunks(id: string): Promise<Chunk[]> {
  return fetchApi<Chunk[]>(`/documents/${id}/chunks`);
}

// ============================================
// QUERY ENDPOINTS
// ============================================

export async function queryKnowledgeBase(
  question: string,
  options?: {
    documentId?: string;
    groupId?: string;
    topK?: number;
    personaId?: string;
  }
): Promise<QueryResult> {
  return fetchApi<QueryResult>("/query", {
    method: "POST",
    body: JSON.stringify({
      question,
      document_id: options?.documentId,
      group_id: options?.groupId,
      top_k: options?.topK || 10,
      persona_id: options?.personaId,
    }),
  });
}

// Streaming query for chat interface
export async function* streamQuery(
  question: string,
  options?: {
    documentId?: string;
    groupId?: string;
    personaId?: string;
  }
): AsyncGenerator<string, void, unknown> {
  const response = await fetch(`${BASE_URL}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      document_id: options?.documentId,
      group_id: options?.groupId,
      persona_id: options?.personaId,
    }),
  });

  if (!response.ok) {
    throw new Error("Query failed");
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    yield decoder.decode(value);
  }
}

// ============================================
// RAPTOR TREE ENDPOINTS
// ============================================

export async function buildRaptorTree(documentId: string): Promise<void> {
  await fetchApi(`/documents/${documentId}/build-tree`, { method: "POST" });
}

export async function getRaptorTree(documentId: string): Promise<{
  layers: number;
  nodesByLayer: Record<number, number>;
}> {
  return fetchApi(`/documents/${documentId}/tree`);
}

// ============================================
// GROUPS ENDPOINTS
// ============================================

export async function getGroups(): Promise<DocumentGroup[]> {
  return fetchApi<DocumentGroup[]>("/groups");
}

export async function createGroup(group: Omit<DocumentGroup, "id">): Promise<DocumentGroup> {
  return fetchApi<DocumentGroup>("/groups", {
    method: "POST",
    body: JSON.stringify(group),
  });
}

export async function updateGroup(id: string, updates: Partial<DocumentGroup>): Promise<DocumentGroup> {
  return fetchApi<DocumentGroup>(`/groups/${id}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export async function deleteGroup(id: string): Promise<void> {
  await fetchApi(`/groups/${id}`, { method: "DELETE" });
}

export async function addDocumentToGroup(groupId: string, documentId: string): Promise<void> {
  await fetchApi(`/groups/${groupId}/documents`, {
    method: "POST",
    body: JSON.stringify({ document_id: documentId }),
  });
}

// ============================================
// HUMAN TASKS ENDPOINTS
// ============================================

export async function getHumanTasks(): Promise<HumanTask[]> {
  return fetchApi<HumanTask[]>("/tasks");
}

export async function completeHumanTask(
  taskId: string,
  files?: File[]
): Promise<HumanTask> {
  const formData = new FormData();
  if (files) {
    files.forEach((file) => formData.append("files", file));
  }

  const response = await fetch(`${BASE_URL}/tasks/${taskId}/complete`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to complete task");
  }

  return response.json();
}

// ============================================
// STATISTICS ENDPOINTS
// ============================================

export async function getStats(): Promise<DatabaseStats> {
  return fetchApi<DatabaseStats>("/stats");
}

// ============================================
// CHAT/CONVERSATION ENDPOINTS
// ============================================

export async function createConversation(personaId?: string): Promise<{ id: string }> {
  return fetchApi("/conversations", {
    method: "POST",
    body: JSON.stringify({ persona_id: personaId }),
  });
}

export async function getConversation(id: string): Promise<{
  id: string;
  messages: Array<{
    role: "user" | "assistant";
    content: string;
    timestamp: string;
  }>;
}> {
  return fetchApi(`/conversations/${id}`);
}

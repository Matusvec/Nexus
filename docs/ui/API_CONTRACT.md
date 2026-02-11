# Nexus API Contract

> This document defines the typed API interfaces between the frontend and backend.
> Backend teams can implement these endpoints; the frontend uses mock implementations in dev mode.

## Base URL

```
NEXT_PUBLIC_API_URL = http://localhost:8000
```

Requests are proxied via Next.js rewrites: `/api/:path*` → `http://localhost:8000/:path*`

---

## Endpoints

### Documents

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| `POST` | `/upload` | `FormData { file, group_id? }` | `Document` | Upload a document |
| `GET` | `/documents` | — | `Document[]` | List all documents |
| `GET` | `/documents/:id` | — | `Document` | Get single document |
| `DELETE` | `/documents/:id` | — | `void` | Delete a document |
| `GET` | `/documents/:id/chunks` | — | `Chunk[]` | Get document chunks |

### Query / Retrieval

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| `POST` | `/query` | `{ question, document_id?, group_id?, top_k?, persona_id? }` | `QueryResult` | Standard RAG query |
| `POST` | `/query/stream` | Same as above | `ReadableStream<string>` | Streaming RAG query |

### RAPTOR Tree

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| `POST` | `/documents/:id/build-tree` | — | `void` | Build RAPTOR tree |
| `GET` | `/documents/:id/tree` | — | `{ layers, nodesByLayer }` | Get tree structure |

### Groups

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| `POST` | `/groups` | `Omit<DocumentGroup, 'id'>` | `DocumentGroup` | Create a group |
| `GET` | `/groups` | — | `DocumentGroup[]` | List all groups |
| `PATCH` | `/groups/:id` | `Partial<DocumentGroup>` | `DocumentGroup` | Update a group |
| `DELETE` | `/groups/:id` | — | `void` | Delete a group |
| `POST` | `/groups/:id/documents` | `{ document_id }` | `void` | Add doc to group |

### Human Tasks

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| `GET` | `/tasks` | — | `HumanTask[]` | List tasks |
| `POST` | `/tasks/:id/complete` | `FormData { files[] }` | `HumanTask` | Complete a task |

### Statistics

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| `GET` | `/stats` | — | `DatabaseStats` | Get system stats |

### Conversations

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| `POST` | `/conversations` | `{ persona_id? }` | `{ id }` | Create conversation |
| `GET` | `/conversations/:id` | — | `Conversation` | Get conversation |

### Agents (Planned)

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| `GET` | `/agents` | — | `Agent[]` | List available agents |
| `POST` | `/agents` | `CreateAgentPayload` | `Agent` | Create custom agent |
| `PATCH` | `/agents/:id` | `Partial<Agent>` | `Agent` | Update agent config |
| `POST` | `/agents/:id/run` | `{ input, context? }` | `AgentRun` | Trigger agent run |
| `GET` | `/agents/:id/runs` | — | `AgentRun[]` | Get run history |

---

## TypeScript Types

All types are defined in `frontend/lib/types.ts`. Key interfaces:

```typescript
interface Document {
  id: string;
  filename: string;
  uploadedAt: string;
  fileSize: number;
  chunkCount: number;
  status: "processing" | "ready" | "error";
  summary?: string;
  groupId?: string;
}

interface QueryResult {
  answer: string;
  sources: MessageSource[];
  queryType: "simple" | "complex" | "exploratory";
  tokensUsed: number;
}

interface Agent {
  id: string;
  personaId: PersonaId;
  name: string;
  role: string;
  description: string;
  tools: string[];
  isCustom: boolean;
}

interface AgentRun {
  id: string;
  agentId: string;
  input: string;
  output?: string;
  status: "running" | "completed" | "failed";
  toolCalls: ToolCall[];
  startedAt: string;
  completedAt?: string;
}
```

---

## Mock Service

The frontend uses mock data when the backend is unavailable. Mock implementations are embedded directly in page components using static data arrays. To connect to a real backend:

1. Set `NEXT_PUBLIC_API_URL` environment variable
2. Ensure backend implements the endpoints above
3. The `lib/api.ts` client handles all HTTP communication

---

## Integration Notes

- **Streaming**: `/query/stream` uses Server-Sent Events (SSE). The frontend reads via `ReadableStream`.
- **File uploads**: Use `FormData` (not JSON) for `/upload` and `/tasks/:id/complete`.
- **Error format**: All errors return `{ detail: string }` with appropriate HTTP status codes.
- **CORS**: Backend must allow `http://localhost:3000` origin in development.

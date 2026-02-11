# Nexus Integration Contracts

> **Version:** 1.0.0  
> **Last updated:** 2026-02-10  
> **Branch:** `feature/agentic-ai-orchestrator`

This document describes the integration contracts between all active workstreams. It defines what each branch expects, provides, and considers breaking.

---

## Active Workstreams

| Branch | Owner | Purpose |
|--------|-------|---------|
| `feature/t-retrieval-hierarchical-rag` | T-Retrieval team | Hierarchical RAG with tree + graph retrieval |
| `feature/hitl-delegation-kernel` | HITL team | Human-in-the-loop governance for agent actions |
| `feature/stunning-frontend` | Frontend team | Web-based UI for Nexus |
| `feature/xr-frontend-prototype` | XR team | VR/AR spatial knowledge interface |
| `feature/agentic-ai-orchestrator` | **This branch** | Agent framework, orchestrator, tools |

---

## What This Branch Expects

### From T-Retrieval Branch

The agentic system uses a **retrieval adapter** (`backend/agents/adapters/retrieval_adapter.py`) that expects the following functions to exist in the T-retrieval modules:

| Module | Function | Signature | Purpose |
|--------|----------|-----------|---------|
| `t_query` | `collapsed_tree_retrieval` | `(query, document_id, top_k) → List[Dict]` | Multi-layer RAG search |
| `t_query` | `extract_query_entities` | `(query) → List[str]` | Extract entities from query |
| `storage` | `get_or_create_collection` | `() → ChromaDB Collection` | Get the chunks collection |
| `storage` | `get_collection_stats` | `() → Dict` | Collection statistics |

**If these modules are not available**, the adapter automatically switches to **mock mode** (`NEXUS_MOCK_RETRIEVAL=1`), returning synthetic results. The agentic system never imports `t_query` or `storage` directly — all access goes through the adapter.

**Expected result format from `collapsed_tree_retrieval`:**
```python
{
    "id": "chunk_123",
    "text": "...",           # or "document": "..."
    "layer": 0,              # 0=base, 1+=summaries
    "score": 0.89,
    "document_id": "doc_1",
}
```

### From HITL Delegation Kernel Branch

The agentic system uses a **HITL adapter** (`backend/agents/adapters/hitl_adapter.py`) that expects:

| Module | Function | Signature | Purpose |
|--------|----------|-----------|---------|
| `hitl_kernel` | `gate_tool_call` | `(agent_id, tool_name, tool_args, permissions) → Dict` | Approve/deny/modify tool calls |
| `hitl_kernel` | `report_result` | `(agent_id, tool_name, success, output_preview) → None` | Audit trail for tool results |

**If these modules are not available**, the adapter operates in **dev mode** (auto-approve everything with warnings). Set `NEXUS_HITL_MODE=strict` to block all side-effect tools instead.

**Expected response format from `gate_tool_call`:**
```python
{
    "approved": True,
    "reason": "Approved by policy",
    "modified_args": None,       # or dict of altered args
}
```

---

## What This Branch Provides

### For Frontend Branch (`feature/stunning-frontend`)

A complete REST API for agent operations, documented in `contracts/agentic/orchestrator_api.py` and `frontend/API_SPECIFICATION.md`.

**Key endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/agents` | List all agents |
| `GET` | `/agents/{id}` | Get agent details |
| `POST` | `/agents` | Create custom agent |
| `PATCH` | `/agents/{id}` | Update custom agent |
| `DELETE` | `/agents/{id}` | Delete custom agent |
| `POST` | `/agents/{id}/chat` | Send message to agent |
| `GET` | `/agents/tools/list` | List available tools |
| `POST` | `/agents/orchestrator/sessions` | Create orchestrator session |
| `POST` | `/agents/orchestrator/sessions/{id}/chat` | Orchestrator chat |
| `POST` | `/agents/orchestrator/sessions/{id}/agent-chat` | Agent-to-agent |
| `GET` | `/agents/status` | System status + feature flags |
| `GET` | `/agents/traces/{id}` | Execution trace for debugging |

**Frontend types** are provided in `frontend/lib/types.ts` with the `AgentInfo`, `AgentTool`, `AgentChatResponse`, `OrchestratorSession`, `OrchestratorMessage` types.

**Frontend components** are in `frontend/components/agents/` — `AgentsPanel`, `AgentChat`, `CreateAgentDialog`, `OrchestratorPanel`.

**Zustand store** is extended in `frontend/lib/store.ts` with `useAgentsStore`.

### For XR Branch (`feature/xr-frontend-prototype`)

The XR branch can consume the **exact same REST API** listed above. No special XR adapter is needed — the API is UI-agnostic.

**For spatial representation of agents:**
- Each agent has `id`, `name`, `role`, `description`, `tools[]`
- Orchestrator sessions have a message log consumable as a conversation transcript
- Execution traces can be visualized as a tree of spans

**Contracts:**
- `contracts/agentic/agent_config_schema.json` — agent configuration validation
- `contracts/agentic/tool_schema.json` — tool definition validation
- `contracts/agentic/orchestrator_api.py` — Python dataclasses for all API shapes

### For All Branches

**Feature flags** allow the system to run without any upstream merges:

| Env Variable | Values | Default | Effect |
|-------------|--------|---------|--------|
| `NEXUS_MOCK_RETRIEVAL` | `0`, `1` | Auto-detect | Force mock RAG results |
| `NEXUS_HITL_MODE` | `auto`, `dev`, `strict`, `kernel` | `auto` | HITL gating mode |

---

## What Changes Would Be Breaking

### Breaking for this branch (if upstream changes):

1. **T-retrieval:** Renaming `collapsed_tree_retrieval` or changing its return format
2. **T-retrieval:** Removing `storage.get_or_create_collection`
3. **HITL kernel:** Changing the `gate_tool_call` function signature or response format

### Breaking for downstream (if this branch changes):

1. Renaming or removing any endpoint in the REST API
2. Changing the structure of `AgentInfo`, `AgentResponsePayload`, or `OrchestratorSession`
3. Removing built-in tool names (`rag_query`, `web_search`, etc.)
4. Changing the `contracts/agentic/*.json` schemas in non-additive ways

---

## Merge Order Recommendation

```
1. feature/t-retrieval-hierarchical-rag    ← merge FIRST (core data layer)
2. feature/agentic-ai-orchestrator         ← merge SECOND (depends on retrieval)
3. feature/hitl-delegation-kernel          ← merge THIRD (gates agent actions)
4. feature/stunning-frontend               ← merge FOURTH (consumes agent API)
5. feature/xr-frontend-prototype           ← merge LAST (consumes same API)
```

**Rationale:**
- Retrieval is the data foundation; everything reads from it
- The agentic system uses retrieval as its knowledge substrate; with the adapter, it works pre-merge but is better post-merge
- HITL kernel is a governance layer that wraps agent tool calls; the no-op shim makes it safe to merge agentic first
- Frontend and XR are pure consumers of the agent API; they should merge after the API is stable
- Frontend and XR are independent of each other

**After each merge, verify:**
1. After T-retrieval merge: Set `NEXUS_MOCK_RETRIEVAL=0`, verify real RAG results in agent chat
2. After agentic merge: Run `python -m agents.demo` to verify agents work end-to-end
3. After HITL merge: Set `NEXUS_HITL_MODE=kernel`, verify tool gating
4. After frontend merge: Verify `/agents` page loads and agent chat works
5. After XR merge: Verify XR can consume `/agents` API

---

## Owned Directories (Do Not Cross-Edit)

| Directory | Owner Branch |
|-----------|-------------|
| `backend/t_retriever.py`, `backend/t_query.py`, `backend/storage.py` | T-retrieval |
| `backend/agents/`, `contracts/agentic/`, `docs/agents/`, `docs/integration/` | **Agentic AI (this branch)** |
| `backend/hitl_kernel/` (if exists) | HITL kernel |
| `frontend/components/` (except `agents/`) | Frontend |
| XR-specific directories | XR prototype |

---

## Testing Without Upstream

```bash
# Run with mock retrieval + dev HITL (works without any merges)
export NEXUS_MOCK_RETRIEVAL=1
export NEXUS_HITL_MODE=dev
cd backend && python -m pytest tests/test_agents.py -v

# Run the demo
cd backend && python -m agents.demo
```

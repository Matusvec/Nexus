# Nexus Agentic AI Architecture

> **Version:** 1.0.0

## Overview

The Nexus agentic AI system provides a multi-agent framework where specialized AI agents collaborate to solve complex tasks. An orchestrator agent coordinates sub-agents, decomposes tasks, and synthesizes results.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 User / Frontend                  │
│         (Web UI / XR / CLI / API)                │
└──────────────────────┬──────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────┐
│              FastAPI Routes                       │
│         (backend/agents/routes.py)               │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│             Agent Registry                        │
│       (backend/agents/registry.py)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Research  │  │   Code   │  │ Planner  │  ...  │
│  │  Agent    │  │  Agent   │  │  Agent   │       │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘       │
│        └──────────────┼──────────────┘            │
│                       │                           │
│        ┌──────────────▼──────────────┐           │
│        │     Orchestrator Agent       │           │
│        │ (multi-agent coordination)   │           │
│        └──────────────┬──────────────┘           │
└───────────────────────┼──────────────────────────┘
                        │
      ┌─────────────────┼──────────────────┐
      │                 │                  │
┌─────▼─────┐  ┌───────▼───────┐  ┌──────▼──────┐
│   Tool     │  │   Retrieval   │  │    HITL     │
│  Registry  │  │   Adapter     │  │   Adapter   │
│ (12 tools) │  │ (mock/real)   │  │ (dev/kernel)│
└────────────┘  └───────┬───────┘  └─────────────┘
                        │
                ┌───────▼───────┐
                │ T-Retrieval   │
                │ RAG Hierarchy │
                └───────────────┘
```

## Built-in Agents

| Agent | Role | Description | Key Tools |
|-------|------|-------------|-----------|
| **Research Agent** | `research` | Deep RAG search with cited answers | `rag_query`, `rag_explain`, `rag_tree_search` |
| **Web Search Agent** | `web_search` | Current information from web + YouTube | `web_search`, `youtube_search`, `rag_query` |
| **Code Agent** | `code` | Code analysis, debugging, math | `repo_inspect`, `rag_query`, `calculate` |
| **Document Agent** | `document` | In-depth document navigation | `rag_query`, `rag_explain`, `document_list` |
| **Planner Agent** | `planner` | Task decomposition | `rag_query`, `workspace_notes` |
| **Synthesis Agent** | `synthesis` | Multi-source answer drafting | `rag_query`, `workspace_notes`, `text_summarize` |

## Tool Registry

All tools follow the schema in `contracts/agentic/tool_schema.json`:

| Tool | Category | Description | Permissions |
|------|----------|-------------|-------------|
| `rag_query` | knowledge | Search RAG hierarchy | No side effects |
| `rag_explain` | knowledge | Explain retrieval path | No side effects |
| `rag_tree_search` | knowledge | Layer-specific RAG search | No side effects |
| `document_list` | knowledge | List all documents | No side effects |
| `document_summary` | knowledge | Document statistics | No side effects |
| `web_search` | web | Web search | Network access |
| `youtube_search` | web | YouTube video search | Network access |
| `text_summarize` | analysis | LLM summarization | No side effects |
| `extract_entities` | analysis | Entity extraction | No side effects |
| `calculate` | utility | Safe math evaluation | No side effects |
| `repo_inspect` | code | Repository file reader | No side effects |
| `workspace_notes` | workspace | Shared agent scratchpad | No side effects |

## Orchestrator

The orchestrator agent:
1. **Decomposes** user tasks into sub-tasks
2. **Selects** the best agent(s) for each sub-task
3. **Dispatches** to agents (in parallel when possible)
4. **Collects** intermediate results via workspace_notes
5. **Reconciles** conflicting information
6. **Synthesizes** a final answer with citations

### Multi-Agent Collaboration Protocol

Agents collaborate via structured messages:

```
User → Orchestrator: "Find motor specs and calculate torque"
  Orchestrator → Planner: decompose task
  Planner → workspace_notes: save plan
  Orchestrator → Research Agent: "find motor specs"
  Orchestrator → Code Agent: "calculate torque from specs"
  Research Agent → workspace_notes: save findings
  Code Agent → workspace_notes: save calculation
  Orchestrator → Synthesis Agent: "combine results"
  Synthesis Agent → User: final answer with citations
```

## Feature Flags

| Flag | Values | Purpose |
|------|--------|---------|
| `NEXUS_MOCK_RETRIEVAL` | `0`/`1` | Use mock or real T-retrieval |
| `NEXUS_HITL_MODE` | `auto`/`dev`/`strict`/`kernel` | HITL gating behavior |

## Observability

Every agent invocation generates a **Trace** (`backend/agents/tracing.py`):
- Trace ID for correlating all spans
- Spans for LLM calls, tool calls, routing decisions
- Input/output summaries
- Duration measurements
- Error tracking

Traces are accessible via `GET /agents/traces/{trace_id}`.

## Custom Agents

Users can create custom agents via `POST /agents`:
```json
{
  "name": "My Physics Tutor",
  "system_prompt": "You explain physics concepts simply...",
  "description": "Helps understand physics",
  "tools": ["rag_query", "calculate", "web_search"],
  "temperature": 0.8
}
```

Custom agents are validated against `contracts/agentic/agent_config_schema.json`.

"""
Nexus Agentic AI — Orchestrator API Contract
=============================================

This module defines the stable API contract that all consumers
(web frontend, XR frontend, CLI, tests) use to interact with
the agentic system.  It is the single source of truth for request /
response shapes.  Frontend and XR branches should code against
these types, NOT against internal implementation classes.

Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────

class AgentRoleEnum(str, Enum):
    RESEARCH   = "research"
    CODE       = "code"
    WEB_SEARCH = "web_search"
    DOCUMENT   = "document"
    PLANNER    = "planner"
    SYNTHESIS  = "synthesis"
    CUSTOM     = "custom"
    ORCHESTRATOR = "orchestrator"


class CollaborationPhase(str, Enum):
    """Structured multi-agent collaboration phases."""
    PROPOSE  = "propose"
    CRITIQUE = "critique"
    REVISE   = "revise"
    FINAL    = "final"


class MessageType(str, Enum):
    USER       = "user"
    AGENT      = "agent"
    TOOL       = "tool"
    DELEGATION = "delegation"
    SUMMARY    = "summary"
    SYSTEM     = "system"


# ── Request contracts ──────────────────────────────────────────────

@dataclass
class AgentChatRequest:
    """POST /agents/{agent_id}/chat"""
    message: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class OrchestratorChatRequest:
    """POST /agents/orchestrator/sessions/{session_id}/chat"""
    message: str
    target_agent_id: Optional[str] = None


@dataclass
class CreateAgentRequest:
    """POST /agents"""
    name: str
    system_prompt: str
    description: str = ""
    role: str = "custom"
    tools: Optional[List[str]] = None
    temperature: float = 0.7
    max_iterations: int = 10
    permissions: Optional[Dict[str, Any]] = None


@dataclass
class CreateSessionRequest:
    """POST /agents/orchestrator/sessions"""
    name: str = "New Session"


@dataclass
class AgentToAgentRequest:
    """POST /agents/orchestrator/sessions/{session_id}/agent-chat"""
    from_agent_id: str
    to_agent_id: str
    message: str


# ── Response contracts ─────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    """A single tool invocation record."""
    tool: str
    args: Dict[str, Any]
    result_preview: str
    success: bool
    duration_ms: float = 0.0


@dataclass
class SourceRecord:
    """A citation to a RAG chunk or web source."""
    chunk_id: str = ""
    document_id: str = ""
    layer: int = 0
    score: float = 0.0
    preview: str = ""
    url: str = ""


@dataclass
class AgentResponsePayload:
    """The response body every agent returns."""
    content: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    sources: List[SourceRecord] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    agent_id: str = ""
    iterations: int = 0
    trace_id: str = ""


@dataclass
class AgentInfo:
    """GET /agents and GET /agents/{id}"""
    id: str
    config: Dict[str, Any]
    created_at: float = 0.0
    message_count: int = 0
    is_custom: bool = False


@dataclass
class SessionMessage:
    """A message inside an orchestrator session."""
    id: str
    sender: str
    sender_name: str
    content: str
    timestamp: float
    message_type: str = "message"
    phase: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorSessionInfo:
    """GET /agents/orchestrator/sessions/{id}"""
    id: str
    name: str
    messages: List[SessionMessage] = field(default_factory=list)
    participating_agents: List[str] = field(default_factory=list)
    created_at: float = 0.0


@dataclass
class OrchestratorChatResponse:
    """POST /agents/orchestrator/sessions/{id}/chat"""
    session_id: str
    response: AgentResponsePayload
    responding_agent: Dict[str, str]
    participating_agents: List[Dict[str, str]] = field(default_factory=list)
    session_messages: List[SessionMessage] = field(default_factory=list)


# ── REST Endpoint Map (for documentation) ─────────────────────────

ENDPOINTS = {
    # Agent CRUD
    "GET    /agents":                                       "List all agents",
    "GET    /agents/{agent_id}":                            "Get agent details",
    "POST   /agents":                                       "Create custom agent",
    "PATCH  /agents/{agent_id}":                            "Update custom agent",
    "DELETE /agents/{agent_id}":                            "Delete custom agent",

    # Agent chat
    "POST   /agents/{agent_id}/chat":                       "Send message to agent",
    "DELETE /agents/{agent_id}/history":                     "Clear agent history",

    # Tools
    "GET    /agents/tools/list":                             "List available tools",

    # Orchestrator sessions
    "POST   /agents/orchestrator/sessions":                  "Create session",
    "GET    /agents/orchestrator/sessions":                  "List sessions",
    "GET    /agents/orchestrator/sessions/{session_id}":     "Get session",
    "POST   /agents/orchestrator/sessions/{session_id}/chat":"Orchestrator chat",
    "POST   /agents/orchestrator/sessions/{session_id}/agent-chat": "Agent-to-agent",

    # Observability
    "GET    /agents/traces/{trace_id}":                      "Get execution trace",
}

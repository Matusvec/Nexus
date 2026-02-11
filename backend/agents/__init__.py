"""
Nexus Agentic AI Framework

Provides:
- Base agent class with tool execution loop
- Tool registry for registering/discovering tools
- Specialized agents (Research, Code, Web, Document, Planner, Synthesis)
- Orchestrator agent for multi-agent coordination
- Custom agent creation and management
- Adapters for T-retrieval, HITL kernel, frontend/XR
- Observability / execution tracing
"""

from agents.base import Agent, AgentConfig, AgentMessage, AgentResponse
from agents.tools import Tool, ToolRegistry, ToolResult
from agents.orchestrator import OrchestratorAgent, OrchestratorSession
from agents.registry import AgentRegistry
from agents.tracing import Trace, TraceSpan, TraceStore, trace_store

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentMessage",
    "AgentResponse",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "OrchestratorAgent",
    "OrchestratorSession",
    "AgentRegistry",
    "Trace",
    "TraceSpan",
    "TraceStore",
    "trace_store",
]

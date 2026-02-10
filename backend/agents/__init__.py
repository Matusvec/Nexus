"""
Nexus Agentic AI Framework

Provides:
- Base agent class with tool execution loop
- Tool registry for registering/discovering tools
- Specialized agents (Research, Code, Web, Document)
- Orchestrator agent for multi-agent coordination
- Custom agent creation and management
"""

from agents.base import Agent, AgentConfig, AgentMessage, AgentResponse
from agents.tools import Tool, ToolRegistry, ToolResult
from agents.orchestrator import OrchestratorAgent, OrchestratorSession
from agents.registry import AgentRegistry

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
]

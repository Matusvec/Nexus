"""
FastAPI endpoints for Nexus Agentic AI system.

Provides REST API for:
- Agent management (list, create, update, delete)
- Agent chat (send messages, get responses)
- Orchestrator sessions (create, send messages, inter-agent chat)
- Tool listing
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.registry import AgentRegistry

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    system_prompt: str = Field(..., min_length=1)
    description: str = ""
    tools: Optional[List[str]] = None
    model: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_iterations: int = Field(default=10, ge=1, le=50)


class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    description: Optional[str] = None
    tools: Optional[List[str]] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_iterations: Optional[int] = None


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = None


class OrchestratorChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    target_agent_id: Optional[str] = None


class CreateSessionRequest(BaseModel):
    name: str = "New Session"


class AgentToAgentRequest(BaseModel):
    from_agent_id: str
    to_agent_id: str
    message: str = Field(..., min_length=1)


# ============================================================================
# ROUTER FACTORY
# ============================================================================


def create_agents_router(registry: AgentRegistry) -> APIRouter:
    """
    Create the FastAPI router for agent endpoints.

    Args:
        registry: The global AgentRegistry instance

    Returns:
        APIRouter with all agent endpoints
    """
    router = APIRouter(prefix="/agents", tags=["agents"])

    # ------------------------------------------------------------------
    # SYSTEM STATUS & OBSERVABILITY (must be before /{agent_id} catch-all)
    # ------------------------------------------------------------------

    @router.get("/status")
    def agent_system_status():
        """Get agentic system status including feature flags."""
        from agents.adapters.retrieval_adapter import MOCK_MODE as ret_mock
        from agents.adapters.hitl_adapter import ACTIVE_MODE as hitl_mode
        return {
            "agents_count": len(registry.agents),
            "tools_count": len(registry.tool_registry.list_tool_names()),
            "retrieval_mode": "mock" if ret_mock else "real",
            "hitl_mode": hitl_mode,
            "orchestrator_ready": registry.orchestrator is not None,
        }

    @router.get("/traces/recent")
    def list_recent_traces():
        """List recent execution traces."""
        from agents.tracing import trace_store
        return trace_store.list_recent()

    @router.get("/traces/{trace_id}")
    def get_trace(trace_id: str):
        """Get a specific execution trace."""
        from agents.tracing import trace_store
        trace = trace_store.get(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace.to_dict()

    @router.get("/tools/list")
    def list_tools():
        """List all available tools."""
        return registry.get_tools()

    # ------------------------------------------------------------------
    # AGENT CRUD
    # ------------------------------------------------------------------

    @router.get("")
    def list_agents():
        """List all available agents."""
        return registry.list_agents()

    @router.get("/{agent_id}")
    def get_agent(agent_id: str):
        """Get details of a specific agent."""
        agent = registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        info = agent.to_dict()
        info["is_custom"] = agent_id in registry._custom_configs
        return info

    @router.post("", status_code=201)
    def create_agent(req: CreateAgentRequest):
        """Create a new custom agent."""
        try:
            agent = registry.create_custom_agent(
                name=req.name,
                system_prompt=req.system_prompt,
                description=req.description,
                tools=req.tools,
                model=req.model,
                temperature=req.temperature,
                max_iterations=req.max_iterations,
            )
            return agent.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.patch("/{agent_id}")
    def update_agent(agent_id: str, req: UpdateAgentRequest):
        """Update a custom agent."""
        updates = req.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        try:
            agent = registry.update_custom_agent(agent_id, updates)
            if not agent:
                raise HTTPException(
                    status_code=404,
                    detail="Agent not found or is a built-in agent (cannot update built-in agents)",
                )
            return agent.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/{agent_id}")
    def delete_agent(agent_id: str):
        """Delete a custom agent."""
        if not registry.delete_custom_agent(agent_id):
            raise HTTPException(
                status_code=404,
                detail="Agent not found or is a built-in agent (cannot delete built-in agents)",
            )
        return {"success": True, "message": f"Agent {agent_id} deleted"}

    # ------------------------------------------------------------------
    # AGENT CHAT
    # ------------------------------------------------------------------

    @router.post("/{agent_id}/chat")
    def chat_with_agent(agent_id: str, req: AgentChatRequest):
        """Send a message to a specific agent."""
        result = registry.chat_with_agent(
            agent_id=agent_id,
            message=req.message,
            context=req.context,
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @router.delete("/{agent_id}/history")
    def clear_agent_history(agent_id: str):
        """Clear an agent's conversation history."""
        agent = registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent.clear_history()
        return {"success": True, "message": "History cleared"}

    # ------------------------------------------------------------------
    # ORCHESTRATOR
    # ------------------------------------------------------------------

    @router.post("/orchestrator/sessions", status_code=201)
    def create_session(req: CreateSessionRequest):
        """Create a new orchestrator session."""
        if not registry.orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        session = registry.orchestrator.create_session(req.name)
        return session.to_dict()

    @router.get("/orchestrator/sessions")
    def list_sessions():
        """List all orchestrator sessions."""
        if not registry.orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        return registry.orchestrator.list_sessions()

    @router.get("/orchestrator/sessions/{session_id}")
    def get_session(session_id: str):
        """Get session details."""
        if not registry.orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        session = registry.orchestrator.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.to_dict()

    @router.post("/orchestrator/sessions/{session_id}/chat")
    def orchestrator_chat(session_id: str, req: OrchestratorChatRequest):
        """Send a message to the orchestrator in a session."""
        if not registry.orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        result = registry.orchestrator.process_message(
            session_id=session_id,
            user_message=req.message,
            target_agent_id=req.target_agent_id,
        )
        return result

    @router.post("/orchestrator/sessions/{session_id}/agent-chat")
    def agent_to_agent(session_id: str, req: AgentToAgentRequest):
        """Facilitate direct agent-to-agent communication."""
        if not registry.orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        result = registry.orchestrator.agent_to_agent_message(
            session_id=session_id,
            from_agent_id=req.from_agent_id,
            to_agent_id=req.to_agent_id,
            message=req.message,
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    return router

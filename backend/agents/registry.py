"""
Agent Registry — manages the lifecycle of all agents in Nexus.

Provides:
- Pre-built specialized agents
- Custom agent creation/update/deletion
- Persistence of agent configurations
- Integration with the orchestrator
"""

from __future__ import annotations

import uuid
import json
import os
from typing import Dict, List, Optional, Any

from agents.base import Agent, AgentConfig, AgentRole
from agents.tools import ToolRegistry, create_default_registry
from agents.orchestrator import OrchestratorAgent


# ============================================================================
# DEFAULT AGENT DEFINITIONS
# ============================================================================

_DEFAULT_AGENTS: List[AgentConfig] = [
    AgentConfig(
        name="Research Agent",
        role=AgentRole.RESEARCH,
        system_prompt=(
            "You are a research specialist AI agent. You excel at finding information "
            "in the knowledge base, synthesizing findings, and providing well-cited answers. "
            "Always search the RAG system before answering questions about documents. "
            "Cite your sources with chunk IDs when possible."
        ),
        description="Searches the RAG knowledge base, synthesizes findings, and provides cited answers",
        tools=["rag_search", "rag_tree_search", "document_list", "document_summary", "text_summarize", "extract_entities"],
    ),
    AgentConfig(
        name="Web Search Agent",
        role=AgentRole.WEB_SEARCH,
        system_prompt=(
            "You are a web search specialist AI agent. You find current information "
            "from the web and YouTube to supplement the knowledge base. "
            "Search for authoritative sources and provide links when available. "
            "Cross-reference web findings with the local knowledge base when relevant."
        ),
        description="Searches the web and YouTube for current information, articles, and videos",
        tools=["web_search", "youtube_search", "rag_search", "text_summarize"],
    ),
    AgentConfig(
        name="Code Agent",
        role=AgentRole.CODE,
        system_prompt=(
            "You are a code specialist AI agent. You help with programming questions, "
            "code analysis, debugging, and technical implementations. "
            "You can search the knowledge base for code-related documents and "
            "perform calculations for algorithm analysis. "
            "Provide code examples with explanations."
        ),
        description="Assists with code analysis, debugging, and technical implementations",
        tools=["rag_search", "calculate", "web_search", "text_summarize"],
    ),
    AgentConfig(
        name="Document Agent",
        role=AgentRole.DOCUMENT,
        system_prompt=(
            "You are a document analysis specialist. You help users understand "
            "their uploaded documents, find specific information, compare documents, "
            "and extract key insights. You navigate the RAG tree hierarchy to find "
            "both high-level summaries and detailed specifics."
        ),
        description="Analyzes documents in-depth, navigates the RAG hierarchy for detailed and summary information",
        tools=["rag_search", "rag_tree_search", "document_list", "document_summary", "extract_entities", "text_summarize"],
    ),
]


class AgentRegistry:
    """
    Central registry that manages all agents and the orchestrator.
    """

    def __init__(self):
        self.tool_registry: ToolRegistry = create_default_registry()
        self.agents: Dict[str, Agent] = {}
        self.orchestrator: Optional[OrchestratorAgent] = None
        self._custom_configs: Dict[str, AgentConfig] = {}

        # Initialize default agents
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Create the default specialized agents."""
        for config in _DEFAULT_AGENTS:
            agent_id = config.role.value
            agent = Agent(agent_id=agent_id, config=config)
            self.agents[agent_id] = agent

        # Create orchestrator
        self.orchestrator = OrchestratorAgent(self.agents, self.tool_registry)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents with their info."""
        result = []
        for agent in self.agents.values():
            info = agent.to_dict()
            info["is_custom"] = agent.id in self._custom_configs
            result.append(info)
        return result

    def create_custom_agent(
        self,
        name: str,
        system_prompt: str,
        description: str = "",
        tools: Optional[List[str]] = None,
        model: str = "",
        temperature: float = 0.7,
        max_iterations: int = 10,
    ) -> Agent:
        """
        Create a new custom agent.

        Args:
            name: Display name for the agent
            system_prompt: System prompt defining the agent's behavior
            description: Short description
            tools: List of tool names the agent can use (None = all tools)
            model: LLM model override
            temperature: LLM temperature
            max_iterations: Max reasoning iterations

        Returns:
            The created Agent instance
        """
        # Validate tool names
        available = set(self.tool_registry.list_tool_names())
        if tools:
            invalid = set(tools) - available
            if invalid:
                raise ValueError(
                    f"Unknown tools: {invalid}. Available: {available}"
                )

        agent_id = f"custom_{uuid.uuid4().hex[:8]}"
        config = AgentConfig(
            name=name,
            role=AgentRole.CUSTOM,
            system_prompt=system_prompt,
            description=description,
            tools=tools or [],
            model=model,
            temperature=temperature,
            max_iterations=max_iterations,
        )

        agent = Agent(agent_id=agent_id, config=config)
        self.agents[agent_id] = agent
        self._custom_configs[agent_id] = config

        # Update orchestrator with new agent
        if self.orchestrator:
            self.orchestrator.agents = self.agents

        return agent

    def update_custom_agent(
        self, agent_id: str, updates: Dict[str, Any]
    ) -> Optional[Agent]:
        """Update a custom agent's configuration."""
        if agent_id not in self._custom_configs:
            return None

        agent = self.agents.get(agent_id)
        if not agent:
            return None

        config_dict = agent.config.to_dict()
        for key in ["name", "system_prompt", "description", "tools", "model", "temperature", "max_iterations"]:
            if key in updates:
                config_dict[key] = updates[key]

        # Validate tool names
        if "tools" in updates and updates["tools"]:
            available = set(self.tool_registry.list_tool_names())
            invalid = set(updates["tools"]) - available
            if invalid:
                raise ValueError(f"Unknown tools: {invalid}")

        new_config = AgentConfig.from_dict(config_dict)
        new_agent = Agent(agent_id=agent_id, config=new_config)
        new_agent.created_at = agent.created_at
        new_agent.conversation_history = agent.conversation_history

        self.agents[agent_id] = new_agent
        self._custom_configs[agent_id] = new_config

        if self.orchestrator:
            self.orchestrator.agents = self.agents

        return new_agent

    def delete_custom_agent(self, agent_id: str) -> bool:
        """Delete a custom agent. Cannot delete built-in agents."""
        if agent_id not in self._custom_configs:
            return False

        del self.agents[agent_id]
        del self._custom_configs[agent_id]

        if self.orchestrator:
            self.orchestrator.agents = self.agents

        return True

    def get_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return self.tool_registry.to_dict()

    def chat_with_agent(
        self,
        agent_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to a specific agent and get a response.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": f"Agent '{agent_id}' not found"}

        response = agent.run(message, self.tool_registry, context=context)
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "response": response.to_dict(),
        }

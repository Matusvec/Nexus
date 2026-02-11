"""
Orchestrator Agent for Nexus.

The orchestrator manages multiple sub-agents, delegates tasks, and
facilitates inter-agent communication in a shared workspace.

Key features:
- Routes user queries to the best-suited agent
- Allows agents to communicate with each other
- Aggregates results from multiple agents
- Maintains a shared conversation log visible to all agents
"""

from __future__ import annotations

import uuid
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from agents.base import Agent, AgentConfig, AgentMessage, AgentResponse, AgentRole


@dataclass
class OrchestratorMessage:
    """A message in the orchestrator's shared workspace."""
    id: str
    sender: str  # agent_id, "user", or "orchestrator"
    sender_name: str
    content: str
    timestamp: float
    message_type: str = "message"  # "message", "tool_result", "delegation", "summary"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "sender_name": self.sender_name,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_type": self.message_type,
            "metadata": self.metadata,
        }


@dataclass
class OrchestratorSession:
    """A multi-agent collaboration session."""
    id: str
    name: str
    messages: List[OrchestratorMessage] = field(default_factory=list)
    participating_agents: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_message(
        self,
        sender: str,
        sender_name: str,
        content: str,
        message_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorMessage:
        msg = OrchestratorMessage(
            id=str(uuid.uuid4()),
            sender=sender,
            sender_name=sender_name,
            content=content,
            timestamp=time.time(),
            message_type=message_type,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        return msg

    def get_transcript(self, last_n: int = 20) -> str:
        """Get a text transcript of recent messages for context."""
        lines = []
        for msg in self.messages[-last_n:]:
            lines.append(f"[{msg.sender_name}]: {msg.content}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "messages": [m.to_dict() for m in self.messages],
            "participating_agents": self.participating_agents,
            "created_at": self.created_at,
        }


class OrchestratorAgent:
    """
    Master orchestrator that coordinates sub-agents.

    The orchestrator:
    1. Analyzes the user's request
    2. Selects which agent(s) to delegate to
    3. Routes the task and collects responses
    4. Facilitates agent-to-agent communication
    5. Synthesizes a final answer
    """

    def __init__(self, agents: Dict[str, Agent], tool_registry: Any):
        self.agents = agents
        self.tool_registry = tool_registry
        self.sessions: Dict[str, OrchestratorSession] = {}
        self.id = "orchestrator"

    def create_session(self, name: str = "New Session") -> OrchestratorSession:
        """Create a new orchestrator session."""
        session = OrchestratorSession(
            id=str(uuid.uuid4()),
            name=name,
        )
        self.sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Optional[OrchestratorSession]:
        return self.sessions.get(session_id)

    def process_message(
        self,
        session_id: str,
        user_message: str,
        target_agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message in an orchestrator session.

        If target_agent_id is provided, route directly to that agent.
        Otherwise, analyze the message and decide which agent(s) to involve.
        """
        session = self.sessions.get(session_id)
        if not session:
            session = self.create_session()

        # Record user message
        session.add_message("user", "User", user_message)

        # If directly targeting an agent
        if target_agent_id and target_agent_id in self.agents:
            return self._delegate_to_agent(session, target_agent_id, user_message)

        # Analyze and route
        routing = self._analyze_and_route(session, user_message)

        if len(routing["agents"]) == 0:
            # Orchestrator handles it directly
            return self._handle_directly(session, user_message)
        elif len(routing["agents"]) == 1:
            # Single agent delegation
            agent_id = routing["agents"][0]
            session.add_message(
                "orchestrator",
                "Orchestrator",
                f"Routing to {self.agents[agent_id].name}: {routing['reasoning']}",
                message_type="delegation",
            )
            return self._delegate_to_agent(session, agent_id, user_message)
        else:
            # Multi-agent collaboration
            return self._multi_agent_collaborate(
                session, routing["agents"], user_message, routing["reasoning"]
            )

    def _analyze_and_route(
        self, session: OrchestratorSession, message: str
    ) -> Dict[str, Any]:
        """Analyze a message and decide which agents should handle it."""
        from gemini_client import generate_content

        agent_descriptions = []
        for aid, agent in self.agents.items():
            tools = ", ".join(agent.config.tools) if agent.config.tools else "all tools"
            agent_descriptions.append(
                f"- {aid}: {agent.name} ({agent.role.value}) - {agent.description}. Tools: {tools}"
            )

        prompt = f"""You are the Nexus Orchestrator. Analyze this user message and decide which agent(s) should handle it.

Available agents:
{chr(10).join(agent_descriptions)}

Recent conversation context:
{session.get_transcript(last_n=5)}

User message: {message}

Respond with EXACTLY this JSON format:
{{"agents": ["agent_id1"], "reasoning": "brief reason"}}

Rules:
- Use 1 agent for focused tasks
- Use 2+ agents for tasks that need multiple expertise areas
- Use empty agents list [] if the orchestrator should answer directly (greetings, simple questions)
- Only use agent IDs from the available list above"""

        response = generate_content(prompt)

        try:
            parsed = json.loads(response)
            if "agents" in parsed:
                # Validate agent IDs
                valid_agents = [a for a in parsed["agents"] if a in self.agents]
                return {
                    "agents": valid_agents,
                    "reasoning": parsed.get("reasoning", ""),
                }
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: try to find agent references in text
        mentioned = [aid for aid in self.agents if aid in response.lower()]
        return {
            "agents": mentioned[:2],
            "reasoning": "Routing based on content analysis",
        }

    def _delegate_to_agent(
        self,
        session: OrchestratorSession,
        agent_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """Delegate a task to a single agent."""
        agent = self.agents[agent_id]

        if agent_id not in session.participating_agents:
            session.participating_agents.append(agent_id)

        # Provide conversation context to the agent
        context = {
            "session_transcript": session.get_transcript(last_n=10),
            "session_id": session.id,
        }

        response = agent.run(message, self.tool_registry, context=context)

        # Record agent response in session
        session.add_message(
            agent_id,
            agent.name,
            response.content,
            metadata={
                "tool_calls": response.tool_calls,
                "sources": response.sources,
                "iterations": response.iterations,
            },
        )

        return {
            "session_id": session.id,
            "response": response.to_dict(),
            "responding_agent": {
                "id": agent_id,
                "name": agent.name,
                "role": agent.role.value,
            },
            "session_messages": [m.to_dict() for m in session.messages[-10:]],
        }

    def _multi_agent_collaborate(
        self,
        session: OrchestratorSession,
        agent_ids: List[str],
        message: str,
        reasoning: str,
    ) -> Dict[str, Any]:
        """Facilitate collaboration between multiple agents."""
        from gemini_client import generate_content

        session.add_message(
            "orchestrator",
            "Orchestrator",
            f"Engaging multiple agents for this task: {', '.join(a for a in agent_ids if a in self.agents)}. {reasoning}",
            message_type="delegation",
        )

        # Filter to valid agent IDs
        valid_ids = [aid for aid in agent_ids if aid in self.agents]

        # Collect responses from each agent
        agent_responses = {}
        for aid in valid_ids:
            agent = self.agents[aid]
            if aid not in session.participating_agents:
                session.participating_agents.append(aid)

            context = {
                "session_transcript": session.get_transcript(last_n=10),
                "session_id": session.id,
                "collaboration_mode": True,
                "other_agents": [a for a in valid_ids if a != aid],
            }
            resp = agent.run(message, self.tool_registry, context=context)
            agent_responses[aid] = resp

            session.add_message(
                aid,
                agent.name,
                resp.content,
                metadata={
                    "tool_calls": resp.tool_calls,
                    "sources": resp.sources,
                },
            )

        # Synthesize final response from all agent inputs
        synthesis_parts = []
        for aid, resp in agent_responses.items():
            name = self.agents[aid].name if aid in self.agents else aid
            synthesis_parts.append(f"{name}'s input:\n{resp.content}")

        synthesis_prompt = f"""You are the Nexus Orchestrator synthesizing inputs from multiple specialist agents.

User's question: {message}

Agent responses:
{chr(10).join(synthesis_parts)}

Synthesize a comprehensive answer that:
1. Combines the best insights from each agent
2. Resolves any contradictions
3. Provides a clear, unified response
4. Credits each agent's contribution where relevant"""

        final_answer = generate_content(synthesis_prompt)

        session.add_message(
            "orchestrator",
            "Orchestrator",
            final_answer,
            message_type="summary",
        )

        # Merge all sources
        all_sources: List[Dict[str, Any]] = []
        all_tool_calls: List[Dict[str, Any]] = []
        for resp in agent_responses.values():
            all_sources.extend(resp.sources)
            all_tool_calls.extend(resp.tool_calls)

        return {
            "session_id": session.id,
            "response": {
                "content": final_answer,
                "tool_calls": all_tool_calls,
                "sources": all_sources,
                "reasoning": [
                    f"{self.agents[aid].name}: {r.reasoning}"
                    for aid, r in agent_responses.items()
                    if aid in self.agents
                ],
                "agent_id": "orchestrator",
                "iterations": sum(r.iterations for r in agent_responses.values()),
            },
            "responding_agent": {
                "id": "orchestrator",
                "name": "Orchestrator",
                "role": "orchestrator",
            },
            "participating_agents": [
                {"id": aid, "name": self.agents[aid].name, "role": self.agents[aid].role.value}
                for aid in valid_ids
            ],
            "session_messages": [m.to_dict() for m in session.messages[-20:]],
        }

    def _handle_directly(
        self, session: OrchestratorSession, message: str
    ) -> Dict[str, Any]:
        """Handle a message directly without delegating."""
        from gemini_client import generate_content

        agent_list = ", ".join(
            f"{a.name} ({a.role.value})" for a in self.agents.values()
        )

        prompt = f"""You are the Nexus Orchestrator. You manage a team of AI agents: {agent_list}.

Respond to the user naturally. If they need specific help, suggest which agent to work with.

User: {message}"""

        response = generate_content(prompt)

        session.add_message("orchestrator", "Orchestrator", response)

        return {
            "session_id": session.id,
            "response": {
                "content": response,
                "tool_calls": [],
                "sources": [],
                "reasoning": ["Handled directly by orchestrator"],
                "agent_id": "orchestrator",
                "iterations": 1,
            },
            "responding_agent": {
                "id": "orchestrator",
                "name": "Orchestrator",
                "role": "orchestrator",
            },
            "session_messages": [m.to_dict() for m in session.messages[-10:]],
        }

    def agent_to_agent_message(
        self,
        session_id: str,
        from_agent_id: str,
        to_agent_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Facilitate direct communication between two agents.
        The 'from' agent sends a message that the 'to' agent processes.
        """
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        from_agent = self.agents.get(from_agent_id)
        to_agent = self.agents.get(to_agent_id)

        if not from_agent or not to_agent:
            return {"error": "Agent not found"}

        # Record the inter-agent message
        session.add_message(
            from_agent_id,
            from_agent.name,
            f"[To {to_agent.name}]: {message}",
            message_type="message",
        )

        # Have the target agent process the message
        context = {
            "session_transcript": session.get_transcript(last_n=10),
            "from_agent": from_agent.name,
            "from_agent_role": from_agent.role.value,
        }
        response = to_agent.run(
            f"Message from {from_agent.name} ({from_agent.role.value}): {message}",
            self.tool_registry,
            context=context,
        )

        session.add_message(
            to_agent_id,
            to_agent.name,
            response.content,
            metadata={"in_reply_to": from_agent_id},
        )

        return {
            "session_id": session_id,
            "from_agent": from_agent_id,
            "to_agent": to_agent_id,
            "response": response.to_dict(),
            "session_messages": [m.to_dict() for m in session.messages[-10:]],
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all orchestrator sessions."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "message_count": len(s.messages),
                "participating_agents": s.participating_agents,
                "created_at": s.created_at,
            }
            for s in self.sessions.values()
        ]

"""
Base Agent class for Nexus Agentic AI system.

Each agent has:
- A system prompt defining its role and personality
- A set of tools it can invoke
- A reasoning loop: think -> select tool -> execute -> reflect -> respond
"""

from __future__ import annotations

import uuid
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class AgentRole(str, Enum):
    RESEARCH = "research"
    CODE = "code"
    WEB_SEARCH = "web_search"
    DOCUMENT = "document"
    CUSTOM = "custom"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentConfig:
    """Configuration for creating an agent."""
    name: str
    role: AgentRole
    system_prompt: str
    description: str = ""
    tools: List[str] = field(default_factory=list)
    model: str = ""
    temperature: float = 0.7
    max_iterations: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "tools": self.tools,
            "model": self.model,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        data = dict(data)
        data["role"] = AgentRole(data["role"])
        return cls(**data)


@dataclass
class AgentMessage:
    """A single message in an agent conversation."""
    role: str  # "user", "assistant", "tool", "system"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.tool_args:
            d["tool_args"] = self.tool_args
        if self.tool_result is not None:
            d["tool_result"] = self.tool_result
        if self.agent_id:
            d["agent_id"] = self.agent_id
        return d


@dataclass
class AgentResponse:
    """Response from an agent's reasoning loop."""
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    agent_id: str = ""
    iterations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "sources": self.sources,
            "reasoning": self.reasoning,
            "agent_id": self.agent_id,
            "iterations": self.iterations,
        }


class Agent:
    """
    Base agent with tool-augmented reasoning loop.

    The agent follows a ReAct-style loop:
    1. Receive user query + conversation history
    2. Reason about what tools to use
    3. Execute tools and collect results
    4. Synthesize a final response

    Tools are called via the Gemini function-calling API.
    """

    def __init__(self, agent_id: str, config: AgentConfig):
        self.id = agent_id
        self.config = config
        self.conversation_history: List[AgentMessage] = []
        self.created_at = time.time()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def role(self) -> AgentRole:
        return self.config.role

    @property
    def description(self) -> str:
        return self.config.description

    def run(
        self,
        user_message: str,
        tool_registry: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Execute the agent's reasoning loop for a user message.

        Args:
            user_message: The user's input
            tool_registry: ToolRegistry instance with available tools
            context: Optional additional context (e.g. document_id)

        Returns:
            AgentResponse with final answer and metadata
        """
        from gemini_client import generate_content

        # Record user message
        self.conversation_history.append(
            AgentMessage(role="user", content=user_message)
        )

        # Build tool descriptions for the prompt
        available_tools = self._get_available_tools(tool_registry)
        tool_descriptions = self._format_tool_descriptions(available_tools)

        reasoning_steps: List[str] = []
        tool_calls_log: List[Dict[str, Any]] = []
        all_sources: List[Dict[str, Any]] = []
        iterations = 0
        accumulated_context = ""

        for iteration in range(self.config.max_iterations):
            iterations = iteration + 1

            # Build prompt for this iteration
            prompt = self._build_prompt(
                user_message=user_message,
                tool_descriptions=tool_descriptions,
                accumulated_context=accumulated_context,
                context=context,
                iteration=iteration,
            )

            # Call LLM
            llm_response = generate_content(prompt)

            # Check if the LLM wants to call a tool
            tool_call = self._parse_tool_call(llm_response)

            if tool_call:
                tool_name = tool_call["tool"]
                tool_args = tool_call["args"]
                reasoning_steps.append(
                    f"Calling tool: {tool_name}({json.dumps(tool_args, default=str)})"
                )

                # Execute tool
                result = tool_registry.execute(tool_name, tool_args)
                tool_calls_log.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result_preview": str(result.output)[:200],
                        "success": result.success,
                    }
                )

                if result.sources:
                    all_sources.extend(result.sources)

                # Add tool result to accumulated context
                accumulated_context += (
                    f"\n\n[Tool: {tool_name}] Result:\n{result.output}"
                )
            else:
                # No tool call — this is the final answer
                reasoning_steps.append("Generating final answer")

                self.conversation_history.append(
                    AgentMessage(role="assistant", content=llm_response)
                )

                return AgentResponse(
                    content=llm_response,
                    tool_calls=tool_calls_log,
                    sources=all_sources,
                    reasoning=reasoning_steps,
                    agent_id=self.id,
                    iterations=iterations,
                )

        # If we exhausted iterations, return what we have
        final_prompt = self._build_final_prompt(
            user_message, accumulated_context, context
        )
        final_answer = generate_content(final_prompt)

        self.conversation_history.append(
            AgentMessage(role="assistant", content=final_answer)
        )

        return AgentResponse(
            content=final_answer,
            tool_calls=tool_calls_log,
            sources=all_sources,
            reasoning=reasoning_steps,
            agent_id=self.id,
            iterations=iterations,
        )

    def _get_available_tools(self, tool_registry: Any) -> List[Any]:
        """Get the tools this agent is allowed to use."""
        if not self.config.tools:
            return tool_registry.list_tools()
        return [
            t for t in tool_registry.list_tools() if t.name in self.config.tools
        ]

    def _format_tool_descriptions(self, tools: List[Any]) -> str:
        """Format tool descriptions for the LLM prompt."""
        if not tools:
            return "No tools available."
        lines = []
        for t in tools:
            params = ", ".join(
                f"{p['name']}: {p['type']}" for p in t.parameters
            )
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        user_message: str,
        tool_descriptions: str,
        accumulated_context: str,
        context: Optional[Dict[str, Any]],
        iteration: int,
    ) -> str:
        """Build the prompt for a single reasoning iteration."""
        history = ""
        for msg in self.conversation_history[-10:]:
            if msg.role in ("user", "assistant"):
                history += f"\n{msg.role.upper()}: {msg.content}"

        ctx_str = ""
        if context:
            ctx_str = f"\nContext: {json.dumps(context, default=str)}"

        return f"""{self.config.system_prompt}

You have access to these tools:
{tool_descriptions}

To call a tool, respond with EXACTLY this JSON format on its own line:
{{"tool": "tool_name", "args": {{"param1": "value1"}}}}

If you do NOT need a tool, respond directly with your answer (no JSON).

Conversation history:{history}
{ctx_str}
{f'Tool results so far:{accumulated_context}' if accumulated_context else ''}

USER: {user_message}

{'Think step by step. If you need more information, call a tool. Otherwise, provide your final answer.' if iteration == 0 else 'Based on the tool results above, either call another tool or provide your final answer.'}"""

    def _build_final_prompt(
        self,
        user_message: str,
        accumulated_context: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build a prompt to force a final answer."""
        ctx_str = ""
        if context:
            ctx_str = f"\nContext: {json.dumps(context, default=str)}"

        return f"""{self.config.system_prompt}

Based on the following information gathered, provide a comprehensive answer.

{ctx_str}
Gathered information:{accumulated_context}

USER: {user_message}

Provide your final answer now. Do not call any tools."""

    @staticmethod
    def _parse_tool_call(response: str) -> Optional[Dict[str, Any]]:
        """
        Parse a tool call from the LLM response.
        Looks for a JSON object with "tool" and "args" keys.
        """
        response = response.strip()
        # Try to parse the entire response as JSON
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find a JSON object in the response using brace matching
        start = response.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(response)):
                if response[i] == "{":
                    depth += 1
                elif response[i] == "}":
                    depth -= 1
                if depth == 0:
                    candidate = response[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
                            return parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break
            start = response.find("{", start + 1)

        return None

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent to dictionary."""
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "created_at": self.created_at,
            "message_count": len(self.conversation_history),
        }

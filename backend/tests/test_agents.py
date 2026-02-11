"""
Tests for the Nexus Agentic AI framework.

Tests the core functionality without requiring external API keys:
- Tool registry operations
- Agent configuration and serialization
- Orchestrator session management
- Tool parsing logic
"""

import json
import math
import os
import pytest
from unittest.mock import patch, MagicMock

# Patch config before importing agents to avoid needing GEMINI_API_KEY
# NOTE: This uses a mock value for testing only — never commit real API keys
import sys
import types

mock_config = types.ModuleType("config")
mock_config.GEMINI_API_KEY = "test-key"  # Mock value for tests only
mock_config.CHROMA_PERSIST_DIR = "/tmp/test_chroma"
mock_config.COLLECTION_NAME = "test_chunks"
mock_config.HOST = "0.0.0.0"
mock_config.PORT = 8000
mock_config.MIN_IMAGE_SIZE_BYTES = 5000
mock_config.PROCESS_IMAGES = False
mock_config.ENTITY_EXTRACTION_MODE = "fast"
mock_config.GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
mock_config.GEMINI_GENERATION_MODEL = "gemini-2.5-flash"
sys.modules["config"] = mock_config

# Mock gemini_client to avoid real API calls
mock_gemini = types.ModuleType("gemini_client")
mock_gemini.generate_content = MagicMock(return_value="Mock LLM response")
mock_gemini.get_embedding = MagicMock(return_value=[0.1] * 768)
mock_gemini.get_embeddings = MagicMock(return_value=[[0.1] * 768])
mock_gemini.client = MagicMock()
sys.modules["gemini_client"] = mock_gemini

# Mock embeddings
mock_embeddings = types.ModuleType("embeddings")
mock_embeddings.get_embedding = MagicMock(return_value=[0.1] * 768)
mock_embeddings.get_embeddings = MagicMock(return_value=[[0.1] * 768])
sys.modules["embeddings"] = mock_embeddings

# Mock storage
mock_storage = types.ModuleType("storage")
mock_storage.get_or_create_collection = MagicMock()
mock_storage.get_collection_stats = MagicMock(return_value={
    "total_chunks": 0,
    "documents": [],
    "layers": [],
    "content_types": {},
})
sys.modules["storage"] = mock_storage

# Mock t_query (owned by T-retrieval branch)
mock_tquery = types.ModuleType("t_query")
mock_tquery.collapsed_tree_retrieval = MagicMock(return_value=[])
mock_tquery.extract_query_entities = MagicMock(return_value=[])
sys.modules["t_query"] = mock_tquery

# Mock t_retriever
mock_tretriever = types.ModuleType("t_retriever")
mock_tretriever.EntityGraph = MagicMock()
mock_tretriever.get_document_graph = MagicMock()
mock_tretriever.load_document_graph = MagicMock()
mock_tretriever.GRAPH_EXPANSION_HOPS = 2
mock_tretriever.GRAPH_EXPANSION_TOP_K = 5
mock_tretriever.TREE_RETRIEVAL_TOP_K = 10
mock_tretriever.HYBRID_ALPHA = 0.5
sys.modules["t_retriever"] = mock_tretriever

# Mock utils
mock_utils = types.ModuleType("utils")
mock_utils.count_tokens = lambda text: len(text) // 4
mock_utils.extract_content_references = MagicMock(return_value={
    "content_types": ["text"],
    "image_refs": [],
    "table_refs": [],
    "has_images": False,
    "has_tables": False,
})
sys.modules["utils"] = mock_utils

# Force mock mode for adapters in tests
os.environ["NEXUS_MOCK_RETRIEVAL"] = "1"
os.environ["NEXUS_HITL_MODE"] = "dev"

# Now import agent modules
from agents.base import Agent, AgentConfig, AgentMessage, AgentResponse, AgentRole
from agents.tools import (
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    create_default_registry,
    _calculate,
)
from agents.orchestrator import OrchestratorAgent, OrchestratorSession
from agents.registry import AgentRegistry


# ============================================================================
# TOOL TESTS
# ============================================================================


class TestToolResult:
    def test_creation(self):
        result = ToolResult(output="test", success=True)
        assert result.output == "test"
        assert result.success is True
        assert result.sources == []

    def test_with_sources(self):
        result = ToolResult(
            output="data",
            success=True,
            sources=[{"id": "1", "text": "src"}],
        )
        assert len(result.sources) == 1


class TestToolParameter:
    def test_to_dict(self):
        param = ToolParameter("query", "string", "Search query")
        d = param.to_dict()
        assert d["name"] == "query"
        assert d["type"] == "string"
        assert d["required"] is True

    def test_optional(self):
        param = ToolParameter("limit", "integer", "Max results", required=False, default=10)
        assert param.required is False
        assert param.default == 10


class TestTool:
    def test_execute(self):
        def my_tool(x: str) -> ToolResult:
            return ToolResult(output=f"got: {x}")

        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=[ToolParameter("x", "string", "Input")],
            fn=my_tool,
        )
        result = tool.execute(x="hello")
        assert result.output == "got: hello"
        assert result.success is True

    def test_to_dict(self):
        tool = Tool(
            name="t",
            description="d",
            parameters=[ToolParameter("a", "string", "arg")],
            fn=lambda a: ToolResult(output=a),
            category="test",
        )
        d = tool.to_dict()
        assert d["name"] == "t"
        assert d["category"] == "test"
        assert len(d["parameters"]) == 1


class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()
        tool = Tool("t1", "test", [], lambda: ToolResult(output="ok"))
        reg.register(tool)
        assert "t1" in reg.list_tool_names()
        assert len(reg.list_tools()) == 1

    def test_execute(self):
        reg = ToolRegistry()
        reg.register(
            Tool("echo", "echo", [ToolParameter("msg", "string", "message")],
                 lambda msg: ToolResult(output=msg))
        )
        result = reg.execute("echo", {"msg": "hello"})
        assert result.output == "hello"
        assert result.success is True

    def test_execute_missing_tool(self):
        reg = ToolRegistry()
        result = reg.execute("nonexistent", {})
        assert result.success is False

    def test_execute_error(self):
        def broken(**kwargs):
            raise RuntimeError("boom")

        reg = ToolRegistry()
        reg.register(Tool("broken", "broken", [], broken))
        result = reg.execute("broken", {})
        assert result.success is False
        assert "boom" in result.output

    def test_default_registry(self):
        reg = create_default_registry()
        names = reg.list_tool_names()
        assert "rag_query" in names
        assert "rag_search" in names  # backward compat alias
        assert "rag_explain" in names
        assert "web_search" in names
        assert "youtube_search" in names
        assert "calculate" in names
        assert "document_list" in names
        assert "rag_tree_search" in names
        assert "text_summarize" in names
        assert "extract_entities" in names
        assert "document_summary" in names
        assert "repo_inspect" in names
        assert "workspace_notes" in names


class TestCalculateTool:
    def test_basic_math(self):
        result = _calculate(expression="2 + 3")
        assert "5" in result.output
        assert result.success is True

    def test_functions(self):
        result = _calculate(expression="sqrt(16)")
        assert "4" in result.output

    def test_invalid(self):
        result = _calculate(expression="import os")
        assert result.success is False


# ============================================================================
# AGENT TESTS
# ============================================================================


class TestAgentConfig:
    def test_serialization(self):
        config = AgentConfig(
            name="Test",
            role=AgentRole.RESEARCH,
            system_prompt="You are a test agent.",
            tools=["rag_search"],
        )
        d = config.to_dict()
        assert d["name"] == "Test"
        assert d["role"] == "research"

        restored = AgentConfig.from_dict(d)
        assert restored.name == "Test"
        assert restored.role == AgentRole.RESEARCH


class TestAgentMessage:
    def test_to_dict(self):
        msg = AgentMessage(role="user", content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert "timestamp" in d

    def test_tool_message(self):
        msg = AgentMessage(
            role="tool",
            content="result",
            tool_name="rag_search",
            tool_args={"query": "test"},
            tool_result="found data",
        )
        d = msg.to_dict()
        assert d["tool_name"] == "rag_search"


class TestAgentResponse:
    def test_to_dict(self):
        resp = AgentResponse(
            content="answer",
            agent_id="test",
            iterations=3,
        )
        d = resp.to_dict()
        assert d["content"] == "answer"
        assert d["iterations"] == 3


class TestAgent:
    def test_creation(self):
        config = AgentConfig(
            name="Test Agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt",
        )
        agent = Agent("test-1", config)
        assert agent.name == "Test Agent"
        assert agent.role == AgentRole.RESEARCH
        assert len(agent.conversation_history) == 0

    def test_to_dict(self):
        config = AgentConfig(
            name="Test Agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt",
        )
        agent = Agent("test-1", config)
        d = agent.to_dict()
        assert d["id"] == "test-1"
        assert d["config"]["name"] == "Test Agent"

    def test_clear_history(self):
        config = AgentConfig(
            name="Test",
            role=AgentRole.RESEARCH,
            system_prompt="Test",
        )
        agent = Agent("t", config)
        agent.conversation_history.append(
            AgentMessage(role="user", content="hi")
        )
        assert len(agent.conversation_history) == 1
        agent.clear_history()
        assert len(agent.conversation_history) == 0

    def test_parse_tool_call_json(self):
        response = '{"tool": "rag_search", "args": {"query": "test"}}'
        result = Agent._parse_tool_call(response)
        assert result is not None
        assert result["tool"] == "rag_search"
        assert result["args"]["query"] == "test"

    def test_parse_tool_call_no_json(self):
        response = "This is a plain text response with no tool call."
        result = Agent._parse_tool_call(response)
        assert result is None

    def test_parse_tool_call_embedded_json(self):
        response = 'Let me search for that.\n{"tool": "web_search", "args": {"query": "AI"}}\n'
        result = Agent._parse_tool_call(response)
        assert result is not None
        assert result["tool"] == "web_search"

    def test_parse_tool_call_nested_args(self):
        response = '{"tool": "complex", "args": {"filter": {"layer": 0, "doc": "test"}}}'
        result = Agent._parse_tool_call(response)
        assert result is not None
        assert result["tool"] == "complex"
        assert result["args"]["filter"]["layer"] == 0

    @patch("gemini_client.generate_content")
    def test_run_direct_answer(self, mock_gen):
        """Test agent returns direct answer when LLM doesn't call tools."""
        mock_gen.return_value = "The answer is 42."

        config = AgentConfig(
            name="Test",
            role=AgentRole.RESEARCH,
            system_prompt="Test",
        )
        agent = Agent("t", config)
        registry = ToolRegistry()

        response = agent.run("What is the answer?", registry)
        assert response.content == "The answer is 42."
        assert response.iterations == 1
        assert len(response.tool_calls) == 0

    @patch("gemini_client.generate_content")
    def test_run_with_tool_call(self, mock_gen):
        """Test agent calls a tool and then responds."""
        mock_gen.side_effect = [
            '{"tool": "echo", "args": {"msg": "hello"}}',
            "The tool returned: hello",
        ]

        config = AgentConfig(
            name="Test",
            role=AgentRole.RESEARCH,
            system_prompt="Test",
            tools=["echo"],
        )
        agent = Agent("t", config)

        registry = ToolRegistry()
        registry.register(
            Tool("echo", "echo", [ToolParameter("msg", "string", "message")],
                 lambda msg: ToolResult(output=msg))
        )

        response = agent.run("Say hello", registry)
        assert response.content == "The tool returned: hello"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["tool"] == "echo"


# ============================================================================
# ORCHESTRATOR TESTS
# ============================================================================


class TestOrchestratorSession:
    def test_create_session(self):
        session = OrchestratorSession(id="s1", name="Test")
        assert session.id == "s1"
        assert len(session.messages) == 0

    def test_add_message(self):
        session = OrchestratorSession(id="s1", name="Test")
        msg = session.add_message("user", "User", "hello")
        assert msg.sender == "user"
        assert len(session.messages) == 1

    def test_transcript(self):
        session = OrchestratorSession(id="s1", name="Test")
        session.add_message("user", "User", "hello")
        session.add_message("agent1", "Agent 1", "hi there")
        transcript = session.get_transcript()
        assert "User" in transcript
        assert "Agent 1" in transcript

    def test_to_dict(self):
        session = OrchestratorSession(id="s1", name="Test")
        session.add_message("user", "User", "test")
        d = session.to_dict()
        assert d["id"] == "s1"
        assert len(d["messages"]) == 1


class TestOrchestratorAgent:
    def setup_method(self):
        self.agents = {}
        for role in [AgentRole.RESEARCH, AgentRole.WEB_SEARCH]:
            config = AgentConfig(
                name=f"{role.value} Agent",
                role=role,
                system_prompt=f"You are a {role.value} agent.",
                description=f"{role.value} specialist",
            )
            self.agents[role.value] = Agent(role.value, config)

        self.registry = ToolRegistry()
        self.orchestrator = OrchestratorAgent(self.agents, self.registry)

    def test_create_session(self):
        session = self.orchestrator.create_session("Test")
        assert session.name == "Test"
        assert session.id in self.orchestrator.sessions

    def test_list_sessions(self):
        self.orchestrator.create_session("S1")
        self.orchestrator.create_session("S2")
        sessions = self.orchestrator.list_sessions()
        assert len(sessions) == 2

    @patch("gemini_client.generate_content")
    def test_handle_directly(self, mock_gen):
        mock_gen.return_value = "Hello! I'm the orchestrator."
        session = self.orchestrator.create_session()
        result = self.orchestrator._handle_directly(session, "hi")
        assert "Hello" in result["response"]["content"]

    @patch("gemini_client.generate_content")
    def test_delegate_to_agent(self, mock_gen):
        mock_gen.return_value = "Research results here."
        session = self.orchestrator.create_session()
        result = self.orchestrator._delegate_to_agent(
            session, "research", "find info about AI"
        )
        assert result["responding_agent"]["id"] == "research"
        assert "research" in session.participating_agents


# ============================================================================
# REGISTRY TESTS
# ============================================================================


class TestAgentRegistry:
    def setup_method(self):
        self.registry = AgentRegistry()

    def test_default_agents(self):
        agents = self.registry.list_agents()
        assert len(agents) >= 6  # research, web_search, code, document, planner, synthesis
        ids = [a["id"] for a in agents]
        assert "research" in ids
        assert "web_search" in ids
        assert "code" in ids
        assert "document" in ids
        assert "planner" in ids
        assert "synthesis" in ids

    def test_orchestrator_created(self):
        assert self.registry.orchestrator is not None

    def test_create_custom_agent(self):
        agent = self.registry.create_custom_agent(
            name="My Agent",
            system_prompt="You help with math.",
            description="Math helper",
            tools=["calculate"],
        )
        assert agent.name == "My Agent"
        assert agent.id.startswith("custom_")

        agents = self.registry.list_agents()
        custom = [a for a in agents if a.get("is_custom")]
        assert len(custom) == 1

    def test_create_custom_invalid_tools(self):
        with pytest.raises(ValueError, match="Unknown tools"):
            self.registry.create_custom_agent(
                name="Bad",
                system_prompt="Test",
                tools=["nonexistent_tool"],
            )

    def test_update_custom_agent(self):
        agent = self.registry.create_custom_agent(
            name="Original",
            system_prompt="Original prompt",
        )
        updated = self.registry.update_custom_agent(
            agent.id, {"name": "Updated"}
        )
        assert updated is not None
        assert updated.name == "Updated"

    def test_update_builtin_fails(self):
        result = self.registry.update_custom_agent("research", {"name": "X"})
        assert result is None

    def test_delete_custom_agent(self):
        agent = self.registry.create_custom_agent(
            name="Temp",
            system_prompt="Temp",
        )
        assert self.registry.delete_custom_agent(agent.id) is True
        assert self.registry.get_agent(agent.id) is None

    def test_delete_builtin_fails(self):
        assert self.registry.delete_custom_agent("research") is False

    def test_get_tools(self):
        tools = self.registry.get_tools()
        assert len(tools) >= 9
        names = [t["name"] for t in tools]
        assert "rag_search" in names

    @patch("gemini_client.generate_content")
    def test_chat_with_agent(self, mock_gen):
        mock_gen.return_value = "Here are the results."
        result = self.registry.chat_with_agent("research", "find something")
        assert result["agent_name"] == "Research Agent"
        assert "results" in result["response"]["content"].lower()

    def test_chat_with_missing_agent(self):
        result = self.registry.chat_with_agent("nonexistent", "hi")
        assert "error" in result


# ============================================================================
# API ROUTE TESTS
# ============================================================================


class TestAgentRoutes:
    """Test the FastAPI routes using TestClient."""

    def setup_method(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from agents.routes import create_agents_router

        self.registry = AgentRegistry()
        self.app = FastAPI()
        self.app.include_router(create_agents_router(self.registry))
        self.client = TestClient(self.app)

    def test_list_agents(self):
        resp = self.client.get("/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 4

    def test_get_agent(self):
        resp = self.client.get("/agents/research")
        assert resp.status_code == 200
        assert resp.json()["config"]["name"] == "Research Agent"

    def test_get_agent_not_found(self):
        resp = self.client.get("/agents/nonexistent")
        assert resp.status_code == 404

    def test_create_agent(self):
        resp = self.client.post("/agents", json={
            "name": "Test Bot",
            "system_prompt": "You are a test bot.",
            "description": "Testing",
            "tools": ["calculate"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["config"]["name"] == "Test Bot"
        assert data["id"].startswith("custom_")

    def test_create_agent_invalid_tools(self):
        resp = self.client.post("/agents", json={
            "name": "Bad Bot",
            "system_prompt": "Test",
            "tools": ["fake_tool"],
        })
        assert resp.status_code == 400

    def test_delete_agent(self):
        # Create then delete
        resp = self.client.post("/agents", json={
            "name": "Temp",
            "system_prompt": "Temp",
        })
        agent_id = resp.json()["id"]

        resp = self.client.delete(f"/agents/{agent_id}")
        assert resp.status_code == 200

    def test_delete_builtin_fails(self):
        resp = self.client.delete("/agents/research")
        assert resp.status_code == 404

    def test_list_tools(self):
        resp = self.client.get("/agents/tools/list")
        assert resp.status_code == 200
        tools = resp.json()
        names = [t["name"] for t in tools]
        assert "rag_search" in names

    def test_create_session(self):
        resp = self.client.post("/agents/orchestrator/sessions", json={
            "name": "Test Session",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Session"

    def test_list_sessions(self):
        self.client.post("/agents/orchestrator/sessions", json={"name": "S1"})
        resp = self.client.get("/agents/orchestrator/sessions")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_system_status(self):
        resp = self.client.get("/agents/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents_count" in data
        assert "retrieval_mode" in data
        assert "hitl_mode" in data
        assert data["retrieval_mode"] == "mock"
        assert data["hitl_mode"] == "dev"

    def test_list_traces(self):
        resp = self.client.get("/agents/traces/recent")
        assert resp.status_code == 200


# ============================================================================
# ADAPTER TESTS
# ============================================================================


class TestRetrievalAdapter:
    """Test the retrieval adapter in mock mode."""

    def test_mock_query(self):
        from agents.adapters.retrieval_adapter import query
        results = query("test query", top_k=3)
        assert len(results) > 0
        assert "text" in results[0]
        assert "score" in results[0]

    def test_mock_explain(self):
        from agents.adapters.retrieval_adapter import explain
        result = explain("test", ["chunk_1", "chunk_2"])
        assert "explanation" in result
        assert "layer_distribution" in result

    def test_mock_list_docs(self):
        from agents.adapters.retrieval_adapter import list_documents
        docs = list_documents()
        assert len(docs) >= 1

    def test_mock_doc_summary(self):
        from agents.adapters.retrieval_adapter import get_document_summary
        summary = get_document_summary("mock_doc")
        assert "chunk_count" in summary


class TestHITLAdapter:
    """Test the HITL adapter in dev mode."""

    def test_dev_mode_approves(self):
        from agents.adapters.hitl_adapter import request_approval, ACTIVE_MODE
        assert ACTIVE_MODE == "dev"
        decision = request_approval("test_agent", "rag_query", {"query": "test"})
        assert bool(decision) is True

    def test_dev_mode_approves_side_effects(self):
        from agents.adapters.hitl_adapter import request_approval
        decision = request_approval(
            "test_agent", "file_write", {"path": "test"},
            {"side_effects": True},
        )
        assert bool(decision) is True  # dev mode auto-approves


# ============================================================================
# TRACING TESTS
# ============================================================================


class TestTracing:
    def test_create_trace(self):
        from agents.tracing import Trace
        trace = Trace(session_id="s1", user_message="test query")
        assert trace.trace_id
        assert trace.session_id == "s1"

    def test_trace_spans(self):
        from agents.tracing import Trace
        trace = Trace(user_message="test")
        span = trace.new_span("rag_query", "tool_call", agent_id="research")
        span.finish(output_summary="found 3 results")
        assert len(trace.spans) == 1
        assert trace.spans[0].duration_ms >= 0

    def test_trace_to_dict(self):
        from agents.tracing import Trace
        trace = Trace(user_message="test")
        trace.set_plan({"steps": ["search", "synthesize"]})
        span = trace.new_span("llm", "llm_call")
        span.finish("got response")
        d = trace.to_dict()
        assert d["plan"] == {"steps": ["search", "synthesize"]}
        assert d["total_spans"] == 1

    def test_trace_store(self):
        from agents.tracing import TraceStore, Trace
        store = TraceStore(max_traces=5)
        for i in range(7):
            store.store(Trace(user_message=f"q{i}"))
        assert len(store.list_recent()) == 5

    def test_trace_summary(self):
        from agents.tracing import Trace
        trace = Trace(user_message="what is gravity?")
        span = trace.new_span("rag_query", "tool_call", agent_id="research")
        span.finish("found info")
        trace.final_output = "Gravity is..."
        summary = trace.summary()
        assert "what is gravity?" in summary
        assert "rag_query" in summary


# ============================================================================
# NEW TOOLS TESTS
# ============================================================================


class TestNewTools:
    def test_rag_explain_tool(self):
        from agents.tools import create_default_registry
        reg = create_default_registry()
        result = reg.execute("rag_explain", {"query": "test", "chunk_ids": "c1,c2"})
        assert result.success
        assert "Explanation" in result.output

    def test_repo_inspect_tool(self):
        from agents.tools import create_default_registry
        reg = create_default_registry()
        result = reg.execute("repo_inspect", {"path": ".", "pattern": ""})
        assert result.success
        assert "Directory" in result.output

    def test_workspace_notes_tool(self):
        from agents.tools import create_default_registry, _workspace_notes as notes
        notes.clear()
        reg = create_default_registry()

        # Set a note
        result = reg.execute("workspace_notes", {"action": "set", "key": "test_key", "value": "hello"})
        assert result.success

        # Get it back
        result = reg.execute("workspace_notes", {"action": "get", "key": "test_key"})
        assert "hello" in result.output

        # List notes
        result = reg.execute("workspace_notes", {"action": "list"})
        assert "test_key" in result.output

        # Delete
        result = reg.execute("workspace_notes", {"action": "delete", "key": "test_key"})
        assert result.success
        notes.clear()

"""
Nexus Agentic AI Demo Script

Run with:  python -m agents.demo

Demonstrates:
  1. Agent system initialization (with feature flag status)
  2. Tool invocations (RAG search, workspace notes, calculate)
  3. Multi-agent collaboration via orchestrator
  4. Execution tracing
  5. Structured output

Works in mock mode (no T-retrieval or HITL kernel needed).
"""

from __future__ import annotations

import os
import sys
import json

# Ensure mock mode for demo
os.environ.setdefault("NEXUS_MOCK_RETRIEVAL", "1")
os.environ.setdefault("NEXUS_HITL_MODE", "dev")

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Patch dependencies for standalone demo
import types

if "config" not in sys.modules:
    mock_config = types.ModuleType("config")
    mock_config.GEMINI_API_KEY = "demo-key"
    mock_config.CHROMA_PERSIST_DIR = "/tmp/demo_chroma"
    mock_config.COLLECTION_NAME = "demo"
    mock_config.GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
    mock_config.GEMINI_GENERATION_MODEL = "gemini-2.5-flash"
    sys.modules["config"] = mock_config

if "gemini_client" not in sys.modules:
    mock_gemini = types.ModuleType("gemini_client")
    _call_counter = {"n": 0}

    def _mock_generate(prompt, **kwargs):
        _call_counter["n"] += 1
        n = _call_counter["n"]
        if n == 1:
            return json.dumps({
                "tool": "rag_query",
                "args": {"query": "motor torque specifications", "top_k": 3},
            })
        elif n == 2:
            return json.dumps({
                "tool": "calculate",
                "args": {"expression": "9.81 * 5"},
            })
        elif n == 3:
            return json.dumps({
                "tool": "workspace_notes",
                "args": {"action": "set", "key": "motor_research", "value": "Found torque specs: 2.5 Nm at 3000 RPM. Force needed: 49.05 N."},
            })
        else:
            return (
                "Based on my research:\n\n"
                "1. **Motor Torque Specs**: The motor provides 2.5 Nm of torque at 3000 RPM "
                "(from RAG search of uploaded documents).\n\n"
                "2. **Force Calculation**: For 5G acceleration, the required force is 49.05 N "
                "(calculated: 9.81 × 5).\n\n"
                "3. **Feasibility**: The motor is suitable for this application, "
                "providing sufficient torque for the 5G acceleration requirement.\n\n"
                "Sources: [mock_chunk_0] Motor specifications document."
            )

    mock_gemini.generate_content = _mock_generate
    mock_gemini.get_embedding = lambda text: [0.1] * 768
    sys.modules["gemini_client"] = mock_gemini

if "embeddings" not in sys.modules:
    mock_emb = types.ModuleType("embeddings")
    mock_emb.get_embedding = lambda text: [0.1] * 768
    mock_emb.get_embeddings = lambda texts: [[0.1] * 768 for _ in texts]
    sys.modules["embeddings"] = mock_emb

if "storage" not in sys.modules:
    from unittest.mock import MagicMock
    mock_storage = types.ModuleType("storage")
    mock_storage.get_or_create_collection = MagicMock()
    mock_storage.get_collection_stats = MagicMock(return_value={"total_chunks": 0, "documents": []})
    sys.modules["storage"] = mock_storage


def main():
    print("=" * 70)
    print("  NEXUS AGENTIC AI DEMO")
    print("=" * 70)
    print()

    # ── Step 1: System Status ─────────────────────────────────
    from agents.adapters.retrieval_adapter import MOCK_MODE as ret_mock
    from agents.adapters.hitl_adapter import ACTIVE_MODE as hitl_mode

    print("📋 System Status:")
    print(f"   Retrieval mode : {'MOCK (no T-retrieval)' if ret_mock else 'REAL (T-retrieval connected)'}")
    print(f"   HITL mode      : {hitl_mode.upper()}")
    print()

    # ── Step 2: Initialize Registry ──────────────────────────
    from agents.registry import AgentRegistry
    from agents.tracing import Trace, trace_store

    registry = AgentRegistry()
    agents = registry.list_agents()
    print(f"🤖 Agents ({len(agents)}):")
    for a in agents:
        role = a["config"]["role"]
        tools = a["config"]["tools"]
        print(f"   [{role:12s}] {a['config']['name']:20s} tools={tools}")
    print()

    # ── Step 3: List Tools ────────────────────────────────────
    tools = registry.get_tools()
    print(f"🔧 Tools ({len(tools)}):")
    for t in tools:
        print(f"   [{t['category']:10s}] {t['name']:20s} {t['description'][:60]}")
    print()

    # ── Step 4: Demo — Agent Chat ─────────────────────────────
    user_query = "What are the motor torque specifications and will the motor handle 5G acceleration?"

    print("─" * 70)
    print(f"💬 User: {user_query}")
    print("─" * 70)
    print()

    # Create trace
    trace = Trace(session_id="demo", user_message=user_query)

    # Reset mock LLM counter for clean demo
    _call_counter["n"] = 0

    # Run the research agent
    agent = registry.get_agent("research")
    assert agent is not None

    response = agent.run(user_query, registry.tool_registry, trace=trace)
    trace.final_output = response.content
    trace_store.store(trace)

    # ── Step 5: Print Results ──────────────────────────────────
    print("🧠 Agent: Research Agent")
    print(f"   Iterations: {response.iterations}")
    print(f"   Tool calls: {len(response.tool_calls)}")
    print()

    print("📞 Tool Calls:")
    for tc in response.tool_calls:
        status = "✅" if tc["success"] else "❌"
        print(f"   {status} {tc['tool']}({json.dumps(tc['args'])})")
        print(f"      → {tc['result_preview'][:100]}")
    print()

    if response.sources:
        print("📚 Sources:")
        for src in response.sources[:5]:
            print(f"   [{src.get('chunk_id', 'unknown')}] "
                  f"layer={src.get('layer', '?')} "
                  f"score={src.get('score', '?')}")
        print()

    print("💡 Reasoning Steps:")
    for step in response.reasoning:
        print(f"   → {step}")
    print()

    print("📝 Final Answer:")
    print("─" * 70)
    for line in response.content.split("\n"):
        print(f"   {line}")
    print("─" * 70)
    print()

    # ── Step 6: Print Trace ────────────────────────────────────
    print("🔍 Execution Trace:")
    print(trace.summary())
    print()

    # ── Step 7: Demo Complete ──────────────────────────────────
    print("=" * 70)
    print("  DEMO COMPLETE")
    print(f"  Trace ID: {trace.trace_id}")
    print(f"  Total spans: {len(trace.spans)}")
    print(f"  Total time: {sum(s.duration_ms for s in trace.spans):.0f}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()

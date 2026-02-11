"""
Observability module — structured tracing for the agentic AI system.

Every orchestrator run produces a Trace object that captures:
- The plan / task decomposition
- Tool calls (inputs, outputs, duration)
- Retrieval explanations
- Orchestration decisions (routing, delegation, synthesis)
- Collaboration phases (propose / critique / revise)

Traces can be retrieved via GET /agents/traces/{trace_id}
and are designed to be consumed by the frontend for debugging.
"""

from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

log = logging.getLogger("nexus.observability")


@dataclass
class TraceSpan:
    """A single span within a trace (one logical unit of work)."""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    span_type: str = ""  # "tool_call", "llm_call", "routing", "collaboration", "synthesis"
    agent_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["TraceSpan"] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def finish(self, output_summary: str = "", error: str = ""):
        self.end_time = time.time()
        if output_summary:
            self.output_summary = output_summary
        if error:
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "span_type": self.span_type,
            "agent_id": self.agent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary[:500],
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
            "error": self.error,
        }


@dataclass
class Trace:
    """A complete execution trace for one orchestrator invocation."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    session_id: str = ""
    user_message: str = ""
    spans: List[TraceSpan] = field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    final_output: str = ""

    def new_span(self, name: str, span_type: str, agent_id: str = "", **kwargs) -> TraceSpan:
        span = TraceSpan(name=name, span_type=span_type, agent_id=agent_id, **kwargs)
        self.spans.append(span)
        log.debug("TRACE [%s] span: %s (%s) agent=%s", self.trace_id, name, span_type, agent_id)
        return span

    def set_plan(self, plan: Dict[str, Any]):
        self.plan = plan
        log.info("TRACE [%s] plan: %s", self.trace_id, plan)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "user_message": self.user_message,
            "plan": self.plan,
            "spans": [s.to_dict() for s in self.spans],
            "final_output": self.final_output[:1000],
            "total_spans": len(self.spans),
            "total_duration_ms": sum(s.duration_ms for s in self.spans),
        }

    def summary(self) -> str:
        """Human-readable trace summary for CLI / logs."""
        lines = [
            f"═══ Trace {self.trace_id} ═══",
            f"Query: {self.user_message[:100]}",
        ]
        if self.plan:
            lines.append(f"Plan: {self.plan}")
        for span in self.spans:
            status = "✓" if not span.error else "✗"
            lines.append(
                f"  {status} [{span.span_type}] {span.name} "
                f"({span.duration_ms:.0f}ms) "
                f"agent={span.agent_id or '-'}"
            )
            if span.input_summary:
                lines.append(f"      in:  {span.input_summary[:120]}")
            if span.output_summary:
                lines.append(f"      out: {span.output_summary[:120]}")
            if span.error:
                lines.append(f"      ERR: {span.error}")
        if self.final_output:
            lines.append(f"Final: {self.final_output[:200]}")
        lines.append("═══ End Trace ═══")
        return "\n".join(lines)


class TraceStore:
    """In-memory store for recent traces. Production would use a DB."""

    def __init__(self, max_traces: int = 200):
        self._traces: Dict[str, Trace] = {}
        self._max = max_traces

    def store(self, trace: Trace):
        if len(self._traces) >= self._max:
            oldest = min(self._traces.values(), key=lambda t: t.created_at)
            del self._traces[oldest.trace_id]
        self._traces[trace.trace_id] = trace

    def get(self, trace_id: str) -> Optional[Trace]:
        return self._traces.get(trace_id)

    def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        traces = sorted(self._traces.values(), key=lambda t: t.created_at, reverse=True)
        return [
            {
                "trace_id": t.trace_id,
                "session_id": t.session_id,
                "user_message": t.user_message[:80],
                "total_spans": len(t.spans),
                "created_at": t.created_at,
            }
            for t in traces[:limit]
        ]


# Global trace store singleton
trace_store = TraceStore()

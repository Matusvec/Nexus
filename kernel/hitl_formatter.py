"""
HITL Formatter - Standardized escalation request format.

When an agent is blocked and must ask the human for help,
this module ensures a strict, structured template is used.
"""

from dataclasses import dataclass, field
from typing import List, Optional


REQUIRED_SECTIONS = [
    "objective",
    "known_facts",
    "unknowns",
    "risks_impact",
    "options",
    "recommendation",
    "questions",
    "next_steps",
]


@dataclass
class EscalationRequest:
    """Structured HITL escalation request."""
    agent_id: str
    objective: str
    known_facts: List[str]
    unknowns: List[str]
    risks_impact: List[str]
    options: List[str]
    recommendation: str
    questions: List[str]
    next_steps: str
    trigger_reasons: List[str] = field(default_factory=list)


def format_escalation(req: EscalationRequest) -> str:
    """Format an EscalationRequest into the HITL protocol template."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"HITL ESCALATION REQUEST — Agent: {req.agent_id}")
    lines.append(f"{'='*60}")
    lines.append("")

    if req.trigger_reasons:
        lines.append("TRIGGER REASONS:")
        for r in req.trigger_reasons:
            lines.append(f"  - {r}")
        lines.append("")

    lines.append(f"1) OBJECTIVE:\n   {req.objective}")
    lines.append("")

    lines.append("2) KNOWN FACTS:")
    for fact in req.known_facts:
        lines.append(f"   - {fact}")
    lines.append("")

    lines.append("3) UNKNOWNS:")
    for unk in req.unknowns:
        lines.append(f"   - {unk}")
    lines.append("")

    lines.append("4) RISKS / IMPACT:")
    for risk in req.risks_impact:
        lines.append(f"   - {risk}")
    lines.append("")

    lines.append("5) OPTIONS:")
    for i, opt in enumerate(req.options, 1):
        label = chr(64 + i)  # A, B, C...
        lines.append(f"   {label}) {opt}")
    lines.append("")

    lines.append(f"6) RECOMMENDATION:\n   {req.recommendation}")
    lines.append("")

    lines.append("7) EXACT QUESTIONS TO HUMAN:")
    for q in req.questions:
        lines.append(f"   ? {q}")
    lines.append("")

    lines.append(f"8) WHAT I WILL DO AFTER YOU ANSWER:\n   {req.next_steps}")
    lines.append("")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


def validate_escalation(req: EscalationRequest) -> List[str]:
    """Validate that all required sections are present and non-empty."""
    errors = []
    if not req.agent_id:
        errors.append("Missing agent_id")
    if not req.objective:
        errors.append("Missing objective")
    if not req.known_facts:
        errors.append("Missing known_facts (must have at least 1)")
    if not req.unknowns:
        errors.append("Missing unknowns (must have at least 1)")
    if not req.risks_impact:
        errors.append("Missing risks_impact (must have at least 1)")
    if not req.options:
        errors.append("Missing options (must have at least 1)")
    if not req.recommendation:
        errors.append("Missing recommendation")
    if not req.questions:
        errors.append("Missing questions (must have at least 1)")
    if not req.next_steps:
        errors.append("Missing next_steps")
    return errors

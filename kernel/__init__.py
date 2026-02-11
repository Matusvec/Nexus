"""
HITL Delegation Kernel - Human-in-the-Loop middleware for agent systems.

Every agent action passes through the kernel_wrapper before execution.
The kernel enforces: PLAN → CHECK → EXECUTE or ESCALATE.
"""

from .kernel_wrapper import KernelWrapper, ActionPlan, KernelResult, Decision
from .risk_engine import RiskEngine, RiskAssessment
from .permission_guard import PermissionGuard, load_manifest, validate_manifest
from .contract_guard import ContractGuard
from .hitl_formatter import EscalationRequest, format_escalation, validate_escalation

__all__ = [
    "KernelWrapper",
    "ActionPlan",
    "KernelResult",
    "Decision",
    "RiskEngine",
    "RiskAssessment",
    "PermissionGuard",
    "load_manifest",
    "validate_manifest",
    "ContractGuard",
    "EscalationRequest",
    "format_escalation",
    "validate_escalation",
]

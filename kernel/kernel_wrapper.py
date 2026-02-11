"""
Kernel Wrapper - The main entrypoint that all agents must pass through.

Enforces: PLAN → CHECK → EXECUTE or ESCALATE
Every agent action goes through this wrapper before executing.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .contract_guard import ContractGuard
from .hitl_formatter import EscalationRequest, format_escalation, validate_escalation
from .permission_guard import PermissionGuard, load_manifest, validate_manifest
from .risk_engine import RiskEngine


class Decision(Enum):
    ALLOW = "ALLOW"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


@dataclass
class ActionPlan:
    """A plan that an agent submits before executing any action."""
    agent_id: str
    objective: str
    assumptions: List[str] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)
    files_read: List[str] = field(default_factory=list)
    files_write: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    action_description: str = ""
    situation: str = ""
    contract_markers: Optional[List[str]] = None


@dataclass
class KernelResult:
    """Result from kernel evaluation."""
    decision: Decision
    reasons: List[str] = field(default_factory=list)
    escalation: Optional[str] = None
    risk_uncertainty: float = 0.0
    risk_impact: float = 0.0


class KernelWrapper:
    """
    The central HITL delegation kernel.

    Usage:
        kernel = KernelWrapper(manifest_path="agents/manifests/rag.json")
        plan = ActionPlan(agent_id="rag", objective="...", ...)
        result = kernel.evaluate(plan)
        if result.decision == Decision.ALLOW:
            # proceed
        else:
            print(result.escalation)
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        manifest: Optional[dict] = None,
        policies_path: Optional[str] = None,
    ):
        if manifest is not None:
            self.manifest = manifest
        elif manifest_path is not None:
            self.manifest = load_manifest(manifest_path)
        else:
            raise ValueError("Either manifest_path or manifest must be provided")

        valid, errors = validate_manifest(self.manifest)
        if not valid:
            raise ValueError(f"Invalid manifest: {errors}")

        self.permission_guard = PermissionGuard(self.manifest)
        self.risk_engine = RiskEngine(policies_path)
        self.contract_guard = ContractGuard(policies_path)
        self.agent_id = self.manifest.get("agent_id", "unknown")

    def evaluate(self, plan: ActionPlan) -> KernelResult:
        """
        Evaluate an action plan: PLAN → CHECK → EXECUTE or ESCALATE.
        Returns a KernelResult with decision and audit trail.
        """
        all_reasons = []

        # Step 1: Permission checks
        perm_ok, perm_reasons = self.permission_guard.check_all(
            files_read=plan.files_read,
            files_write=plan.files_write,
            tools=plan.tools,
            action_description=plan.action_description,
            situation=plan.situation,
        )
        all_reasons.extend(perm_reasons)

        if not perm_ok:
            return KernelResult(
                decision=Decision.BLOCK,
                reasons=all_reasons,
                escalation=self._build_escalation(plan, all_reasons, "Permission denied"),
            )

        # Step 2: Contract guard
        all_files = plan.files_write  # Only writes matter for contracts
        contract_ok, contract_violations = self.contract_guard.check_plan(
            files_to_modify=all_files,
            plan_text=plan.action_description,
            markers=plan.contract_markers,
        )
        if not contract_ok:
            all_reasons.extend(contract_violations)
            return KernelResult(
                decision=Decision.BLOCK,
                reasons=all_reasons,
                escalation=self._build_escalation(
                    plan, all_reasons, "Contract guard violation"
                ),
            )

        # Step 3: Risk assessment
        assessment = self.risk_engine.assess(
            assumptions=plan.assumptions,
            verification_steps=plan.verification_steps,
            files_touched=plan.files_read + plan.files_write,
            actions=plan.actions,
        )
        all_reasons.extend(assessment.reasons)

        if assessment.hard_stops:
            all_reasons.extend(assessment.hard_stops)
            return KernelResult(
                decision=Decision.ESCALATE,
                reasons=all_reasons,
                escalation=self._build_escalation(plan, all_reasons, "Risk hard-stop"),
                risk_uncertainty=assessment.uncertainty_score,
                risk_impact=assessment.impact_score,
            )

        # Step 4: Threshold checks
        agent_thresholds = self.manifest.get("escalate_if", {}).get(
            "soft_thresholds", {}
        )
        assessment = self.risk_engine.check_thresholds(assessment, agent_thresholds)
        all_reasons.extend(
            [r for r in assessment.reasons if r not in all_reasons]
        )

        if not assessment.allowed:
            all_reasons.extend(assessment.hard_stops)
            return KernelResult(
                decision=Decision.ESCALATE,
                reasons=all_reasons,
                escalation=self._build_escalation(
                    plan, all_reasons, "Threshold exceeded"
                ),
                risk_uncertainty=assessment.uncertainty_score,
                risk_impact=assessment.impact_score,
            )

        # All checks passed
        return KernelResult(
            decision=Decision.ALLOW,
            reasons=all_reasons,
            risk_uncertainty=assessment.uncertainty_score,
            risk_impact=assessment.impact_score,
        )

    def _build_escalation(
        self, plan: ActionPlan, reasons: List[str], trigger: str
    ) -> str:
        """Build a formatted HITL escalation message."""
        req = EscalationRequest(
            agent_id=plan.agent_id,
            objective=plan.objective,
            known_facts=[f"Action: {plan.action_description}"] if plan.action_description else ["Plan submitted"],
            unknowns=plan.assumptions if plan.assumptions else ["No specific unknowns identified"],
            risks_impact=reasons[:5],
            options=["Proceed with human approval", "Modify plan", "Abort"],
            recommendation="Request human review before proceeding",
            questions=["Should I proceed with this action?", "Are there additional constraints?"],
            next_steps="Will proceed according to human decision",
            trigger_reasons=[trigger] + reasons[:3],
        )
        return format_escalation(req)

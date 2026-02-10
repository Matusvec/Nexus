"""
Risk Engine - Computes uncertainty and impact scores for agent actions.

Provides deterministic, auditable risk assessment based on:
- Number of assumptions in the plan
- Missing verification steps
- Cross-context changes (multiple bounded contexts)
- Modifications to high-risk directories

All decisions are logged with explanations.
"""

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class RiskAssessment:
    """Result of a risk assessment with full audit trail."""
    uncertainty_score: float
    impact_score: float
    reasons: List[str] = field(default_factory=list)
    hard_stops: List[str] = field(default_factory=list)
    allowed: bool = True

    @property
    def needs_escalation(self) -> bool:
        return len(self.hard_stops) > 0


def _load_policies(policies_path: Optional[str] = None) -> dict:
    if policies_path is None:
        policies_path = str(Path(__file__).parent / "policies.yaml")
    with open(policies_path, "r") as f:
        return yaml.safe_load(f)


class RiskEngine:
    """Computes risk scores for an action plan."""

    def __init__(self, policies_path: Optional[str] = None):
        self.policies = _load_policies(policies_path)
        self.weights = self.policies.get("risk_weights", {})

    def assess(
        self,
        assumptions: List[str],
        verification_steps: List[str],
        files_touched: List[str],
        actions: List[str],
        bounded_contexts: Optional[List[str]] = None,
    ) -> RiskAssessment:
        reasons = []
        hard_stops = []
        uncertainty = 0.0
        impact = 0.0

        # 1. Assumptions
        n_assumptions = len(assumptions)
        max_before_escalation = self.policies["global"]["max_assumptions_before_escalation"]
        assumption_weight = self.weights.get("assumptions", 0.15)
        assumption_contrib = min(n_assumptions * assumption_weight, 1.0)
        uncertainty += assumption_contrib
        if n_assumptions > 0:
            reasons.append(
                f"assumptions={n_assumptions} (weight={assumption_weight}, contrib={assumption_contrib:.2f})"
            )
        if n_assumptions >= max_before_escalation:
            hard_stops.append(
                f"Too many assumptions ({n_assumptions} >= {max_before_escalation}): escalation required"
            )

        # 2. Missing verification
        missing_weight = self.weights.get("missing_verification", 0.25)
        if not verification_steps:
            uncertainty += missing_weight
            reasons.append(f"No verification steps provided (contrib={missing_weight:.2f})")

        # 3. Cross-context
        if bounded_contexts is None:
            # Infer contexts from file paths (top-level dirs)
            bounded_contexts = list({Path(f).parts[0] for f in files_touched if f})
        cross_weight = self.weights.get("cross_context", 0.20)
        if len(bounded_contexts) > 1:
            cross_contrib = min(len(bounded_contexts) * cross_weight * 0.5, cross_weight)
            uncertainty += cross_contrib
            reasons.append(
                f"Cross-context changes ({bounded_contexts}): contrib={cross_contrib:.2f}"
            )

        # 4. High-risk directories
        high_risk_patterns = self.policies.get("high_risk_paths", [])
        hr_weight = self.weights.get("high_risk_dir", 0.40)
        matched_hr = []
        for f in files_touched:
            for pattern in high_risk_patterns:
                if fnmatch.fnmatch(f, pattern):
                    matched_hr.append(f)
                    break
        if matched_hr:
            hr_contrib = min(len(matched_hr) * hr_weight * 0.25, hr_weight)
            impact += hr_contrib
            reasons.append(f"High-risk files touched ({matched_hr}): impact={hr_contrib:.2f}")

        # 5. Destructive actions
        destructive = self.policies.get("destructive_actions", [])
        for action in actions:
            for d in destructive:
                if d.lower() in action.lower():
                    hard_stops.append(f"Destructive action detected: '{action}' matches '{d}'")
                    impact = max(impact, 1.0)

        # 6. Contract paths
        contract_patterns = self.policies.get("contract_paths", [])
        contract_files = []
        for f in files_touched:
            for pattern in contract_patterns:
                if fnmatch.fnmatch(f, pattern):
                    contract_files.append(f)
                    break
        if contract_files:
            hard_stops.append(
                f"Contract files modified without marker: {contract_files}"
            )
            impact = max(impact, 0.9)

        # Clamp scores
        uncertainty = min(uncertainty, 1.0)
        impact = min(impact, 1.0)

        allowed = len(hard_stops) == 0
        return RiskAssessment(
            uncertainty_score=round(uncertainty, 4),
            impact_score=round(impact, 4),
            reasons=reasons,
            hard_stops=hard_stops,
            allowed=allowed,
        )

    def check_thresholds(
        self, assessment: RiskAssessment, agent_thresholds: Optional[dict] = None
    ) -> RiskAssessment:
        """Apply threshold checks and add soft-stop escalations."""
        thresholds = agent_thresholds or {}
        unc_threshold = thresholds.get(
            "uncertainty_score",
            self.policies["global"]["default_uncertainty_threshold"],
        )
        imp_threshold = thresholds.get(
            "impact_score",
            self.policies["global"]["default_impact_threshold"],
        )

        if assessment.uncertainty_score >= unc_threshold:
            assessment.hard_stops.append(
                f"Uncertainty {assessment.uncertainty_score:.2f} >= threshold {unc_threshold}"
            )
            assessment.allowed = False

        if (
            assessment.impact_score >= imp_threshold
            and not assessment.hard_stops
        ):
            assessment.reasons.append(
                f"Impact {assessment.impact_score:.2f} >= threshold {imp_threshold} (soft warning)"
            )

        return assessment

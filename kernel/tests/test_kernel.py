"""
Tests for the HITL Delegation Kernel.

Unit tests:
- Manifest parsing + schema validation
- Permission checks (allowed/blocked)
- HITL formatter output matches required template
- Contract guard triggers when modifying contracts without marker
- Risk engine scoring

Scenario tests:
1) Ambiguous requirements → MUST ESCALATE
2) Attempt to modify contracts/openapi.yaml without contract-change → BLOCK
3) Low-risk change within agent scope → ALLOW
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure kernel package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kernel.permission_guard import PermissionGuard, validate_manifest, load_manifest
from kernel.risk_engine import RiskEngine, RiskAssessment
from kernel.contract_guard import ContractGuard
from kernel.hitl_formatter import (
    EscalationRequest,
    format_escalation,
    validate_escalation,
    REQUIRED_SECTIONS,
)
from kernel.kernel_wrapper import KernelWrapper, ActionPlan, Decision, KernelResult


# ─── Fixtures ────────────────────────────────────────────────

KERNEL_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = KERNEL_DIR.parent / "agents" / "manifests"
POLICIES_PATH = str(KERNEL_DIR / "policies.yaml")


def _sample_manifest():
    return {
        "agent_id": "test_agent",
        "description": "Test agent for unit tests",
        "allowed_file_globs": {
            "read": ["backend/**/*.py", "docs/**"],
            "write": ["backend/test_module/**"],
        },
        "allowed_tools": ["file_edit", "llm_call"],
        "can": ["edit test files", "run tests"],
        "cannot": ["modify security config", "delete databases"],
        "requires_human_for": ["breaking API changes", "security decisions"],
        "escalate_if": {
            "hard_stops": ["ambiguous requirements", "destructive actions"],
            "soft_thresholds": {"uncertainty_score": 0.35, "impact_score": 0.7},
        },
        "verification_steps": ["python -m pytest"],
    }


# ─── Unit Tests: Manifest Validation ─────────────────────────

class TestManifestValidation:
    def test_valid_manifest(self):
        valid, errors = validate_manifest(_sample_manifest())
        assert valid is True
        assert errors == []

    def test_missing_required_key(self):
        manifest = _sample_manifest()
        del manifest["agent_id"]
        valid, errors = validate_manifest(manifest)
        assert valid is False
        assert any("agent_id" in e for e in errors)

    def test_missing_file_globs_mode(self):
        manifest = _sample_manifest()
        del manifest["allowed_file_globs"]["write"]
        valid, errors = validate_manifest(manifest)
        assert valid is False
        assert any("write" in e for e in errors)

    def test_missing_escalate_if_keys(self):
        manifest = _sample_manifest()
        del manifest["escalate_if"]["hard_stops"]
        valid, errors = validate_manifest(manifest)
        assert valid is False
        assert any("hard_stops" in e for e in errors)

    def test_invalid_allowed_tools_type(self):
        manifest = _sample_manifest()
        manifest["allowed_tools"] = "not_a_list"
        valid, errors = validate_manifest(manifest)
        assert valid is False
        assert any("allowed_tools" in e for e in errors)

    def test_all_agent_manifests_valid(self):
        """Validate all shipped agent manifests conform to schema."""
        for manifest_file in MANIFESTS_DIR.glob("*.json"):
            with open(manifest_file) as f:
                manifest = json.load(f)
            valid, errors = validate_manifest(manifest)
            assert valid is True, f"{manifest_file.name}: {errors}"


# ─── Unit Tests: Permission Guard ────────────────────────────

class TestPermissionGuard:
    def setup_method(self):
        self.guard = PermissionGuard(_sample_manifest())

    def test_file_read_allowed(self):
        ok, reason = self.guard.check_file_access("backend/utils/helper.py", "read")
        assert ok is True

    def test_file_read_denied(self):
        ok, reason = self.guard.check_file_access("frontend/app/page.tsx", "read")
        assert ok is False
        assert "DENIED" in reason

    def test_file_write_allowed(self):
        ok, reason = self.guard.check_file_access("backend/test_module/foo.py", "write")
        assert ok is True

    def test_file_write_denied(self):
        ok, reason = self.guard.check_file_access("backend/main.py", "write")
        assert ok is False

    def test_tool_allowed(self):
        ok, reason = self.guard.check_tool("file_edit")
        assert ok is True

    def test_tool_denied(self):
        ok, reason = self.guard.check_tool("db_query")
        assert ok is False

    def test_capability_blocked(self):
        ok, reason = self.guard.check_capability("delete databases now")
        assert ok is False

    def test_capability_allowed(self):
        ok, reason = self.guard.check_capability("edit test files")
        assert ok is True

    def test_requires_human(self):
        needs, reason = self.guard.requires_human("this involves breaking API changes")
        assert needs is True

    def test_no_human_needed(self):
        needs, reason = self.guard.requires_human("minor formatting fix")
        assert needs is False

    def test_check_all_allowed(self):
        ok, reasons = self.guard.check_all(
            files_read=["backend/utils/helper.py"],
            files_write=["backend/test_module/foo.py"],
            tools=["file_edit"],
            action_description="edit test files",
        )
        assert ok is True

    def test_check_all_blocked(self):
        ok, reasons = self.guard.check_all(
            files_write=["frontend/app/page.tsx"],
            tools=["db_query"],
        )
        assert ok is False


# ─── Unit Tests: Risk Engine ─────────────────────────────────

class TestRiskEngine:
    def setup_method(self):
        self.engine = RiskEngine(POLICIES_PATH)

    def test_low_risk(self):
        result = self.engine.assess(
            assumptions=[],
            verification_steps=["run tests"],
            files_touched=["backend/utils.py"],
            actions=["edit file"],
        )
        assert result.uncertainty_score < 0.35
        assert result.allowed is True

    def test_high_assumptions(self):
        result = self.engine.assess(
            assumptions=["assume A", "assume B", "assume C"],
            verification_steps=["run tests"],
            files_touched=["backend/utils.py"],
            actions=["edit file"],
        )
        assert result.needs_escalation is True
        assert any("assumptions" in s.lower() for s in result.hard_stops)

    def test_no_verification(self):
        result = self.engine.assess(
            assumptions=[],
            verification_steps=[],
            files_touched=["backend/utils.py"],
            actions=["edit file"],
        )
        assert result.uncertainty_score >= 0.25

    def test_destructive_action(self):
        result = self.engine.assess(
            assumptions=[],
            verification_steps=["test"],
            files_touched=["backend/db.py"],
            actions=["DROP TABLE users"],
        )
        assert result.needs_escalation is True
        assert any("destructive" in s.lower() for s in result.hard_stops)

    def test_contract_file(self):
        result = self.engine.assess(
            assumptions=[],
            verification_steps=["test"],
            files_touched=["contracts/openapi.yaml"],
            actions=["update spec"],
        )
        assert result.needs_escalation is True

    def test_high_risk_directory(self):
        result = self.engine.assess(
            assumptions=[],
            verification_steps=["test"],
            files_touched=["security/auth.py"],
            actions=["review"],
        )
        assert result.impact_score > 0

    def test_threshold_check(self):
        result = self.engine.assess(
            assumptions=["a1", "a2"],
            verification_steps=[],
            files_touched=["backend/a.py", "frontend/b.ts"],
            actions=["cross-context edit"],
        )
        result = self.engine.check_thresholds(result, {"uncertainty_score": 0.35})
        # With 2 assumptions + no verification + cross-context, uncertainty should be high
        assert result.uncertainty_score > 0


# ─── Unit Tests: Contract Guard ──────────────────────────────

class TestContractGuard:
    def setup_method(self):
        self.guard = ContractGuard(POLICIES_PATH)

    def test_non_contract_file(self):
        assert self.guard.is_contract_file("backend/utils.py") is False

    def test_contract_file_match(self):
        assert self.guard.is_contract_file("contracts/openapi.yaml") is True

    def test_security_file_match(self):
        assert self.guard.is_contract_file("security/auth.py") is True

    def test_plan_allowed_no_contracts(self):
        ok, violations = self.guard.check_plan(
            files_to_modify=["backend/utils.py"],
            plan_text="simple edit",
        )
        assert ok is True
        assert violations == []

    def test_plan_blocked_contract_no_marker(self):
        ok, violations = self.guard.check_plan(
            files_to_modify=["contracts/openapi.yaml"],
            plan_text="update API spec",
        )
        assert ok is False
        assert len(violations) > 0
        assert "CONTRACT-CHANGE" in violations[0]

    def test_plan_allowed_contract_with_marker(self):
        ok, violations = self.guard.check_plan(
            files_to_modify=["contracts/openapi.yaml"],
            plan_text="update API spec",
            markers=["CONTRACT-CHANGE"],
        )
        assert ok is True

    def test_plan_allowed_contract_marker_in_text(self):
        ok, violations = self.guard.check_plan(
            files_to_modify=["contracts/openapi.yaml"],
            plan_text="update API spec CONTRACT-CHANGE approved",
        )
        assert ok is True


# ─── Unit Tests: HITL Formatter ──────────────────────────────

class TestHITLFormatter:
    def _sample_request(self):
        return EscalationRequest(
            agent_id="test_agent",
            objective="Test objective",
            known_facts=["Fact 1", "Fact 2"],
            unknowns=["Unknown 1"],
            risks_impact=["Risk 1"],
            options=["Option A", "Option B"],
            recommendation="Go with Option A",
            questions=["Should I proceed?"],
            next_steps="Will implement Option A if approved",
            trigger_reasons=["Test trigger"],
        )

    def test_format_contains_all_sections(self):
        req = self._sample_request()
        output = format_escalation(req)
        assert "1) OBJECTIVE:" in output
        assert "2) KNOWN FACTS:" in output
        assert "3) UNKNOWNS:" in output
        assert "4) RISKS / IMPACT:" in output
        assert "5) OPTIONS:" in output
        assert "6) RECOMMENDATION:" in output
        assert "7) EXACT QUESTIONS TO HUMAN:" in output
        assert "8) WHAT I WILL DO AFTER YOU ANSWER:" in output

    def test_format_contains_agent_id(self):
        req = self._sample_request()
        output = format_escalation(req)
        assert "test_agent" in output

    def test_format_contains_trigger_reasons(self):
        req = self._sample_request()
        output = format_escalation(req)
        assert "TRIGGER REASONS:" in output
        assert "Test trigger" in output

    def test_format_contains_questions(self):
        req = self._sample_request()
        output = format_escalation(req)
        assert "? Should I proceed?" in output

    def test_format_options_labeled(self):
        req = self._sample_request()
        output = format_escalation(req)
        assert "A) Option A" in output
        assert "B) Option B" in output

    def test_validate_valid_request(self):
        req = self._sample_request()
        errors = validate_escalation(req)
        assert errors == []

    def test_validate_missing_objective(self):
        req = self._sample_request()
        req.objective = ""
        errors = validate_escalation(req)
        assert any("objective" in e.lower() for e in errors)

    def test_validate_empty_known_facts(self):
        req = self._sample_request()
        req.known_facts = []
        errors = validate_escalation(req)
        assert any("known_facts" in e.lower() for e in errors)

    def test_validate_missing_questions(self):
        req = self._sample_request()
        req.questions = []
        errors = validate_escalation(req)
        assert any("questions" in e.lower() for e in errors)


# ─── Scenario Tests ──────────────────────────────────────────

class TestScenarios:
    """Integration scenarios testing full kernel flow."""

    def _make_kernel(self):
        return KernelWrapper(manifest=_sample_manifest(), policies_path=POLICIES_PATH)

    def test_scenario_ambiguous_requirements_must_escalate(self):
        """Scenario 1: Ambiguous requirements → MUST ESCALATE."""
        kernel = self._make_kernel()
        plan = ActionPlan(
            agent_id="test_agent",
            objective="Implement user auth (unclear which method)",
            assumptions=[
                "Assume OAuth is preferred",
                "Assume session-based is acceptable",
                "Assume no SSO required",
            ],
            verification_steps=[],
            files_read=["backend/utils/helper.py"],
            files_write=["backend/test_module/auth.py"],
            tools=["file_edit"],
            actions=["create auth module"],
            action_description="edit test files for auth",
            situation="ambiguous requirements with multiple approaches",
        )
        result = kernel.evaluate(plan)
        assert result.decision in (Decision.ESCALATE, Decision.BLOCK), (
            f"Expected ESCALATE or BLOCK, got {result.decision}"
        )

    def test_scenario_contract_modification_without_marker_blocked(self):
        """Scenario 2: Modify contracts/openapi.yaml without CONTRACT-CHANGE → BLOCK."""
        kernel = self._make_kernel()
        plan = ActionPlan(
            agent_id="test_agent",
            objective="Update API specification",
            assumptions=[],
            verification_steps=["validate spec"],
            files_read=["backend/utils/helper.py"],
            files_write=["contracts/openapi.yaml"],
            tools=["file_edit"],
            actions=["update openapi spec"],
            action_description="edit test files",
        )
        result = kernel.evaluate(plan)
        # Should be blocked because contracts/openapi.yaml isn't in write globs
        # AND contract guard blocks it without marker
        assert result.decision in (Decision.BLOCK, Decision.ESCALATE), (
            f"Expected BLOCK or ESCALATE, got {result.decision}"
        )

    def test_scenario_low_risk_within_scope_allowed(self):
        """Scenario 3: Low-risk change within agent scope → ALLOW."""
        kernel = self._make_kernel()
        plan = ActionPlan(
            agent_id="test_agent",
            objective="Fix typo in test utility",
            assumptions=[],
            verification_steps=["python -m pytest"],
            files_read=["backend/utils/helper.py"],
            files_write=["backend/test_module/utils.py"],
            tools=["file_edit"],
            actions=["fix typo"],
            action_description="edit test files",
        )
        result = kernel.evaluate(plan)
        assert result.decision == Decision.ALLOW, (
            f"Expected ALLOW, got {result.decision}: {result.reasons}"
        )

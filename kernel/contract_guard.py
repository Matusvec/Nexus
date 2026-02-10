"""
Contract Guard - Enforces rules around contract-sensitive files.

Blocks or requires explicit CONTRACT-CHANGE markers when modifying:
- contracts/ (OpenAPI, DB schema, ML IO, design tokens)
- security/ sensitive files
- retrieval/ core RAG hierarchy code
"""

import fnmatch
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


def _load_policies(policies_path: Optional[str] = None) -> dict:
    if policies_path is None:
        policies_path = str(Path(__file__).parent / "policies.yaml")
    with open(policies_path, "r") as f:
        return yaml.safe_load(f)


class ContractGuard:
    """Validates that contract-sensitive files are not modified without markers."""

    def __init__(self, policies_path: Optional[str] = None):
        self.policies = _load_policies(policies_path)
        self.contract_paths = self.policies.get("contract_paths", [])
        self.marker = self.policies.get("contract_change_marker", "CONTRACT-CHANGE")

    def is_contract_file(self, file_path: str) -> bool:
        """Check if a file matches any contract path pattern."""
        for pattern in self.contract_paths:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False

    def check_plan(
        self,
        files_to_modify: List[str],
        plan_text: str = "",
        markers: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Check if a plan that modifies contract files has the required marker.
        Returns (allowed, list_of_violations).
        """
        violations = []
        contract_files = [f for f in files_to_modify if self.is_contract_file(f)]

        if not contract_files:
            return True, []

        has_marker = False
        if markers and self.marker in markers:
            has_marker = True
        if self.marker in plan_text:
            has_marker = True

        if not has_marker:
            for cf in contract_files:
                violations.append(
                    f"Contract file '{cf}' modified without '{self.marker}' marker"
                )

        return len(violations) == 0, violations

    def check_git_diff(
        self, repo_path: str = ".", markers: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check git diff for contract file modifications without marker.
        Returns (allowed, violations).
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_path,
            )
            changed_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except Exception as e:
            return False, [f"Failed to get git diff: {e}"]

        contract_files = [f for f in changed_files if self.is_contract_file(f)]
        if not contract_files:
            return True, []

        # Check commit message for marker
        has_marker = False
        if markers and self.marker in markers:
            has_marker = True

        try:
            msg_result = subprocess.run(
                ["git", "log", "-1", "--format=%B"],
                capture_output=True,
                text=True,
                cwd=repo_path,
            )
            if self.marker in msg_result.stdout:
                has_marker = True
        except Exception:
            pass

        if has_marker:
            return True, []

        violations = [
            f"Contract file '{cf}' changed without '{self.marker}' marker in commit"
            for cf in contract_files
        ]
        return len(violations) == 0, violations

"""
Permission Guard - Validates agent actions against manifest permissions.

Checks file access (read/write globs), tool usage, and capability boundaries.
All decisions are deterministic and return explanations.
"""

import fnmatch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_manifest(manifest_path: str) -> dict:
    """Load and return an agent manifest from JSON file."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def validate_manifest(manifest: dict, schema_path: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Validate a manifest dict against the schema. Returns (valid, errors)."""
    errors = []
    required_keys = ["agent_id", "allowed_file_globs", "allowed_tools", "can", "cannot",
                     "requires_human_for", "escalate_if"]
    for key in required_keys:
        if key not in manifest:
            errors.append(f"Missing required key: {key}")

    if "allowed_file_globs" in manifest:
        fg = manifest["allowed_file_globs"]
        if not isinstance(fg, dict):
            errors.append("allowed_file_globs must be an object")
        else:
            for mode in ("read", "write"):
                if mode not in fg:
                    errors.append(f"allowed_file_globs missing '{mode}' key")
                elif not isinstance(fg[mode], list):
                    errors.append(f"allowed_file_globs.{mode} must be an array")

    if "allowed_tools" in manifest and not isinstance(manifest["allowed_tools"], list):
        errors.append("allowed_tools must be an array")

    if "escalate_if" in manifest:
        esc = manifest["escalate_if"]
        if not isinstance(esc, dict):
            errors.append("escalate_if must be an object")
        else:
            if "hard_stops" not in esc:
                errors.append("escalate_if missing 'hard_stops'")
            if "soft_thresholds" not in esc:
                errors.append("escalate_if missing 'soft_thresholds'")

    return len(errors) == 0, errors


class PermissionGuard:
    """Checks whether an agent's planned action is permitted by its manifest."""

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.agent_id = manifest.get("agent_id", "unknown")

    def check_file_access(self, file_path: str, mode: str = "read") -> Tuple[bool, str]:
        """Check if agent can access file_path in given mode (read/write)."""
        globs = self.manifest.get("allowed_file_globs", {}).get(mode, [])
        for pattern in globs:
            if fnmatch.fnmatch(file_path, pattern):
                return True, f"Agent '{self.agent_id}' allowed {mode} on '{file_path}' (matched '{pattern}')"
        return False, f"Agent '{self.agent_id}' DENIED {mode} on '{file_path}': no matching glob in manifest"

    def check_tool(self, tool_name: str) -> Tuple[bool, str]:
        """Check if agent is allowed to use a tool."""
        allowed = self.manifest.get("allowed_tools", [])
        if tool_name in allowed:
            return True, f"Agent '{self.agent_id}' allowed tool '{tool_name}'"
        return False, f"Agent '{self.agent_id}' DENIED tool '{tool_name}': not in allowed_tools"

    def check_capability(self, action_description: str) -> Tuple[bool, str]:
        """Check action against can/cannot lists (substring match)."""
        cannot = self.manifest.get("cannot", [])
        for rule in cannot:
            if rule.lower() in action_description.lower():
                return False, f"Agent '{self.agent_id}' BLOCKED: action matches cannot rule '{rule}'"

        can = self.manifest.get("can", [])
        for rule in can:
            if rule.lower() in action_description.lower():
                return True, f"Agent '{self.agent_id}' allowed: action matches can rule '{rule}'"

        return True, f"Agent '{self.agent_id}' allowed: no explicit denial found"

    def requires_human(self, situation: str) -> Tuple[bool, str]:
        """Check if situation requires human intervention per manifest."""
        rh = self.manifest.get("requires_human_for", [])
        for rule in rh:
            if rule.lower() in situation.lower():
                return True, f"Agent '{self.agent_id}' requires human for: '{rule}'"
        return False, f"Agent '{self.agent_id}' does not require human for this situation"

    def check_all(
        self,
        files_read: Optional[List[str]] = None,
        files_write: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        action_description: str = "",
        situation: str = "",
    ) -> Tuple[bool, List[str]]:
        """Run all permission checks. Returns (allowed, list_of_reasons)."""
        reasons = []
        allowed = True

        for f in (files_read or []):
            ok, reason = self.check_file_access(f, "read")
            reasons.append(reason)
            if not ok:
                allowed = False

        for f in (files_write or []):
            ok, reason = self.check_file_access(f, "write")
            reasons.append(reason)
            if not ok:
                allowed = False

        for t in (tools or []):
            ok, reason = self.check_tool(t)
            reasons.append(reason)
            if not ok:
                allowed = False

        if action_description:
            ok, reason = self.check_capability(action_description)
            reasons.append(reason)
            if not ok:
                allowed = False

        if situation:
            needs_human, reason = self.requires_human(situation)
            reasons.append(reason)
            if needs_human:
                allowed = False

        return allowed, reasons

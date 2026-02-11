"""
HITL (Human-in-the-Loop) Adapter — bridges the agentic system to the
HITL Delegation Kernel.

When the HITL kernel branch modules are present, this adapter delegates
gating decisions to the real kernel.  When absent (pre-merge), it runs
in "dev mode" — all tool calls are auto-approved with loud warnings.

Feature flag:  NEXUS_HITL_MODE controls behavior:
    "kernel"  → use real HITL kernel (default when kernel is present)
    "dev"     → auto-approve everything with warnings
    "strict"  → block all side-effect tools (for safety testing)
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("nexus.adapters.hitl")

# ── Feature flag ──────────────────────────────────────────────────

_HITL_MODE = os.getenv("NEXUS_HITL_MODE", "auto")


def _kernel_available() -> bool:
    """Check whether the HITL Delegation Kernel is importable."""
    try:
        import hitl_kernel  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_mode() -> str:
    if _HITL_MODE == "auto":
        return "kernel" if _kernel_available() else "dev"
    return _HITL_MODE


ACTIVE_MODE = _resolve_mode()

if ACTIVE_MODE == "dev":
    log.warning(
        "⚠️  HITL ADAPTER: Running in DEV mode (no HITL kernel). "
        "All tool calls auto-approved. NOT SAFE FOR PRODUCTION. "
        "Set NEXUS_HITL_MODE=strict to block side-effect tools."
    )
elif ACTIVE_MODE == "strict":
    log.warning(
        "🔒 HITL ADAPTER: Running in STRICT mode. "
        "All side-effect tools are BLOCKED."
    )
else:
    log.info("✅ HITL ADAPTER: Using real HITL Delegation Kernel.")


# ── Approval decision ─────────────────────────────────────────────

class ApprovalDecision:
    """Represents a HITL gating decision."""

    def __init__(self, approved: bool, reason: str = "", modified_args: Optional[Dict] = None):
        self.approved = approved
        self.reason = reason
        self.modified_args = modified_args  # Kernel may alter args for safety

    def __bool__(self):
        return self.approved


# ── Public interface ──────────────────────────────────────────────


def request_approval(
    agent_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_permissions: Optional[Dict[str, Any]] = None,
) -> ApprovalDecision:
    """
    Request HITL approval for a tool call.

    Called by the agent execution loop before every tool invocation.
    The HITL kernel (when present) may:
      - approve as-is
      - approve with modified args (e.g. scoping file paths)
      - block the call
      - queue for human review

    Args:
        agent_id: Which agent is requesting
        tool_name: Tool being invoked
        tool_args: Arguments to the tool
        tool_permissions: Tool permission metadata from tool_schema.json

    Returns:
        ApprovalDecision
    """
    permissions = tool_permissions or {}

    if ACTIVE_MODE == "kernel":
        return _kernel_approve(agent_id, tool_name, tool_args, permissions)
    elif ACTIVE_MODE == "strict":
        return _strict_approve(agent_id, tool_name, tool_args, permissions)
    else:
        return _dev_approve(agent_id, tool_name, tool_args, permissions)


def report_tool_result(
    agent_id: str,
    tool_name: str,
    success: bool,
    output_preview: str = "",
) -> None:
    """
    Report tool execution result back to the HITL kernel for auditing.
    No-op in dev mode.
    """
    if ACTIVE_MODE == "kernel":
        try:
            import hitl_kernel
            hitl_kernel.report_result(agent_id, tool_name, success, output_preview)
        except Exception as e:
            log.error("HITL result reporting failed: %s", e)
    else:
        log.debug(
            "HITL audit [%s]: agent=%s tool=%s success=%s",
            ACTIVE_MODE, agent_id, tool_name, success,
        )


# ── Mode implementations ──────────────────────────────────────────


def _kernel_approve(agent_id, tool_name, tool_args, permissions):
    try:
        import hitl_kernel
        result = hitl_kernel.gate_tool_call(agent_id, tool_name, tool_args, permissions)
        return ApprovalDecision(
            approved=result.get("approved", False),
            reason=result.get("reason", ""),
            modified_args=result.get("modified_args"),
        )
    except Exception as e:
        log.error("HITL kernel gating failed, falling back to deny: %s", e)
        return ApprovalDecision(False, f"Kernel error: {e}")


def _strict_approve(agent_id, tool_name, tool_args, permissions):
    has_side_effects = permissions.get("side_effects", False)
    requires_network = permissions.get("network_access", False)

    if has_side_effects:
        log.warning("🔒 STRICT: Blocked %s (side effects)", tool_name)
        return ApprovalDecision(False, f"Tool '{tool_name}' blocked in strict mode (has side effects)")
    if requires_network:
        log.warning("🔒 STRICT: Blocked %s (network access)", tool_name)
        return ApprovalDecision(False, f"Tool '{tool_name}' blocked in strict mode (network access)")

    return ApprovalDecision(True, "Approved (no side effects/network)")


def _dev_approve(agent_id, tool_name, tool_args, permissions):
    has_side_effects = permissions.get("side_effects", False)
    if has_side_effects:
        log.warning(
            "⚠️  DEV MODE: Auto-approving tool '%s' WITH side effects "
            "(agent: %s). This would require HITL approval in production.",
            tool_name, agent_id,
        )
    return ApprovalDecision(True, "Auto-approved (dev mode)")

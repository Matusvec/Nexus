# HITL Delegation Kernel

A modular Human-in-the-Loop middleware that enforces safe execution boundaries for all agents in the Nexus system.

## Overview

The kernel acts as a mandatory checkpoint for every agent action:

```
Agent Action → PLAN → CHECK → EXECUTE or ESCALATE
```

### Core Components

| Module | Purpose |
|--------|---------|
| `kernel_wrapper.py` | Main entrypoint — evaluates action plans |
| `risk_engine.py` | Computes uncertainty/impact scores |
| `permission_guard.py` | Validates file/tool/capability permissions |
| `contract_guard.py` | Guards contract-sensitive files |
| `hitl_formatter.py` | Formats escalation requests |
| `policies.yaml` | Global thresholds and rules |
| `manifest_schema.json` | JSON Schema for agent manifests |

### CLI Tools

| Script | Purpose |
|--------|---------|
| `kernel/check_plan` | Validate an action plan against policies |
| `kernel/check_changes` | Check git diff for contract violations |

## Quick Start

### 1. Adopt the Wrapper in an Agent

```python
from kernel import KernelWrapper, ActionPlan, Decision

# Load kernel with agent manifest
kernel = KernelWrapper(manifest_path="agents/manifests/rag.json")

# Before any action, submit a plan
plan = ActionPlan(
    agent_id="rag",
    objective="Index new document into RAG pipeline",
    assumptions=["Document is PDF format"],
    verification_steps=["Run retrieval test"],
    files_read=["backend/storage.py"],
    files_write=["backend/t_retriever.py"],
    tools=["llm_call", "db_query"],
    actions=["store embeddings"],
    action_description="Index document into vector store",
)

result = kernel.evaluate(plan)

if result.decision == Decision.ALLOW:
    # Proceed with action
    execute_action()
elif result.decision == Decision.ESCALATE:
    # Show escalation to human
    print(result.escalation)
elif result.decision == Decision.BLOCK:
    # Action is blocked
    print("BLOCKED:", result.reasons)
```

### 2. Write a New Agent Manifest

Create a JSON file in `agents/manifests/`:

```json
{
  "agent_id": "my_agent",
  "description": "What this agent does",
  "allowed_file_globs": {
    "read": ["backend/**/*.py", "docs/**"],
    "write": ["backend/my_module/**"]
  },
  "allowed_tools": ["file_edit", "llm_call"],
  "can": ["edit backend code", "run tests"],
  "cannot": ["modify security config", "delete databases"],
  "requires_human_for": ["breaking API changes", "security decisions"],
  "escalate_if": {
    "hard_stops": ["ambiguous requirements", "destructive actions"],
    "soft_thresholds": {
      "uncertainty_score": 0.35,
      "impact_score": 0.7
    }
  },
  "verification_steps": ["python -m pytest tests/"]
}
```

### 3. Contract Guard

The contract guard prevents modifications to sensitive files without explicit markers:

**Protected paths** (configurable in `policies.yaml`):
- `contracts/**` — OpenAPI specs, DB schemas, ML IO definitions
- `security/**` — Security configurations
- `retrieval/**` — Core RAG hierarchy code

**How it works:**
- If an agent's plan includes writing to a contract path, the `CONTRACT-CHANGE` marker must be present
- Without the marker, the action is blocked with a detailed violation message

**CLI usage:**
```bash
# Check a plan
kernel/check_plan --plan plan.json --manifest agents/manifests/rag.json

# Check git changes
kernel/check_changes --repo-path .
```

### 4. Run Tests

```bash
# Install test dependencies
pip install pyyaml pytest

# Run all kernel tests
python -m pytest kernel/tests/ -v

# Run specific test
python -m pytest kernel/tests/test_kernel.py -v
```

## Architecture

```
kernel/
├── __init__.py              # Package exports
├── kernel_wrapper.py        # Main PLAN→CHECK→EXECUTE/ESCALATE
├── risk_engine.py           # Uncertainty + impact scoring
├── permission_guard.py      # File/tool/capability checks
├── contract_guard.py        # Contract-sensitive file guards
├── hitl_formatter.py        # Escalation message formatter
├── policies.yaml            # Global configuration
├── manifest_schema.json     # Agent manifest schema
├── check_plan               # CLI: validate action plan
├── check_changes            # CLI: check git diff
├── HITL_PROTOCOL.md         # Escalation protocol specification
├── README.md                # This file
└── tests/
    ├── __init__.py
    └── test_kernel.py       # Unit + scenario tests

agents/
└── manifests/               # Per-agent capability manifests
    ├── architecture.json
    ├── database.json
    ├── api.json
    └── ...
```

## Design Principles

1. **Not prompt-only** — Implemented as code middleware, not just instructions
2. **Deterministic & auditable** — Every decision includes explanations
3. **Modular** — No agent-specific logic in kernel; specialization lives in manifests
4. **Non-invasive** — Wraps existing architecture without rewriting it

## Limitations & Next Steps

- Risk scoring uses heuristic weights (not ML-based) — suitable for v1
- Contract guard git diff check requires git to be available
- Future: integrate with orchestrator's tool dispatch loop
- Future: persist audit logs to database
- Future: dynamic threshold adjustment based on historical data

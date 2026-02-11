# HITL Escalation Protocol

## Purpose

This document defines the **strict template** that every agent must use when it is blocked by the HITL Delegation Kernel and needs to request human input.

No agent may use vague requests like "need info" or "help me". All escalations must follow this structured format.

## Escalation Template

When an agent is blocked, it must produce the following sections:

### 1) Objective
What the agent is trying to accomplish. One clear sentence.

### 2) Known Facts
Bullet list of verified information the agent has.

### 3) Unknowns
Bullet list of information the agent does NOT have but needs.

### 4) Risks / Impact
Bullet list of potential consequences if the action proceeds incorrectly.

### 5) Options (A/B/C)
At least two concrete options the human can choose from.

### 6) Recommendation
The agent's recommended option with brief justification.

### 7) Exact Questions to Human
Specific, answerable questions. Not vague. Each question must be actionable.

**Good:** "Should I use PostgreSQL or SQLite for the session store?"
**Bad:** "What database should I use?"

### 8) What I Will Do After You Answer
Concrete next steps the agent will take once the human responds.

## Example Escalation

```
============================================================
HITL ESCALATION REQUEST — Agent: database
============================================================

TRIGGER REASONS:
  - Contract guard violation
  - Contract file 'contracts/schema.sql' modified without 'CONTRACT-CHANGE' marker

1) OBJECTIVE:
   Add a new 'sessions' table to support user authentication

2) KNOWN FACTS:
   - Current schema has 'users' and 'documents' tables
   - Authentication feature requires session storage
   - PostgreSQL is the production database

3) UNKNOWNS:
   - Session expiry policy (TTL)
   - Whether to use JWT or server-side sessions
   - Maximum concurrent sessions per user

4) RISKS / IMPACT:
   - Schema migration affects production database
   - Wrong session strategy could create security vulnerabilities
   - Breaking change if other agents depend on current schema

5) OPTIONS:
   A) Add sessions table with server-side sessions (standard approach)
   B) Use JWT tokens with no server-side state (stateless)
   C) Defer to security team for session architecture decision

6) RECOMMENDATION:
   Option A — server-side sessions provide better revocation control

7) EXACT QUESTIONS TO HUMAN:
   ? What should the session TTL be (e.g., 24 hours, 7 days)?
   ? Should we limit concurrent sessions per user? If yes, what limit?
   ? Is this a CONTRACT-CHANGE approved modification to the DB schema?

8) WHAT I WILL DO AFTER YOU ANSWER:
   Will create the migration SQL with the specified TTL, add the
   CONTRACT-CHANGE marker, update the schema documentation, and
   run the migration test suite before applying.
============================================================
```

## Validation Rules

The HITL formatter (`hitl_formatter.py`) enforces:
- All 8 sections must be present
- `known_facts`, `unknowns`, `risks_impact`, `options`, `questions` must each have ≥1 item
- `agent_id`, `objective`, `recommendation`, `next_steps` must be non-empty strings

## When Escalation is Required

### Hard Stops (always escalate)
- Ambiguous requirements (≥2 plausible interpretations)
- Security-sensitive design or secrets/PII
- Breaking interface changes (OpenAPI/DB schema/design tokens/ML IO)
- Destructive actions (data deletion, DROP, rm -rf)
- Inability to verify (no tests available)
- Touching paths outside manifest permissions

### Soft Stops (escalate if threshold exceeded)
- `uncertainty_score >= 0.35` (configurable per agent)
- `impact_score >= 0.7` with weak verification
- Conflicts between agents' outputs

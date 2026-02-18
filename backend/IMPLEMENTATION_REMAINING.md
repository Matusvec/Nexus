# Nexus Backend — Remaining Implementation Prompt

> **Context:** This is a handoff document for completing the Nexus PM pipeline backend. The core pipeline (evidence → problems → embeddings → clusters → proposals → tasks → roadmap) is functional end-to-end. What remains is **missing endpoints, production hardening, infrastructure upgrades, and quality expansion.** Read the companion `strategybackend.md` for full architectural context.

---

## Table of Contents

1. [Current State Summary](#current-state-summary)
2. [Task 1: Missing API Endpoints (~7 endpoints)](#task-1-missing-api-endpoints)
3. [Task 2: Wire Authentication to All Routes](#task-2-wire-authentication-to-all-routes)
4. [Task 3: Citation Verification Pipeline](#task-3-citation-verification-pipeline)
5. [Task 4: Prompt Registry & Versioned Prompt Files](#task-4-prompt-registry--versioned-prompt-files)
6. [Task 5: Expand Eval Harness Golden Set](#task-5-expand-eval-harness-golden-set)
7. [Task 6: HDBSCAN + UMAP Clustering Upgrade](#task-6-hdbscan--umap-clustering-upgrade)
8. [Task 7: Multi-Provider LLM Abstraction](#task-7-multi-provider-llm-abstraction)
9. [Task 8: Celery + Redis Job Queue (Optional/Deferred)](#task-8-celery--redis-job-queue)
10. [Task 9: Observability & Metrics Endpoint](#task-9-observability--metrics-endpoint)
11. [Task 10: Docker Production Hardening](#task-10-docker-production-hardening)
12. [Codebase Patterns & Conventions](#codebase-patterns--conventions)
13. [File Reference Map](#file-reference-map)

---

## Current State Summary

### What Works (Don't Touch Unless Fixing Bugs)

| Component | Status | Files |
|-----------|--------|-------|
| Evidence CRUD + chunking | ✅ Complete | `services/evidence_service.py`, `routers/evidence.py`, `utils/chunking.py` |
| Problem extraction (LLM) | ✅ Complete | `services/extraction_service.py`, `routers/jobs.py` |
| Embedding generation | ✅ Complete | `services/embeddings_service.py` |
| Similarity search | ✅ Complete | `services/problems_service.py`, `routers/problems.py` |
| Problem stats/filtering | ✅ Complete | `services/problems_service.py`, `routers/problems.py` |
| Threshold clustering | ✅ Complete | `services/cluster_service.py`, `routers/clusters.py` |
| Cluster summarization (LLM) | ✅ Complete | `services/cluster_service.py` |
| Proposal generation (LLM) | ✅ Complete | `services/proposal_service.py` |
| Task tree generation (LLM) | ✅ Complete | `services/task_tree_service.py`, `routers/tasks.py` |
| Prioritization scoring | ✅ Complete | `services/prioritization_service.py` |
| Roadmap ranking | ✅ Complete | `routers/clusters.py` |
| LLM cost tracking (in-memory + DB) | ✅ Complete | `llm/client.py`, `models/jobs.py` (LLMCallLog) |
| Rate limiting middleware | ✅ Active | `middleware/rate_limit.py` |
| All 12+ ORM models | ✅ Complete | `models/` directory |
| All 4 Alembic migrations | ✅ Complete | `alembic/versions/` |
| Health check | ✅ Complete | `main.py` |

### What's Missing (Your Work Items)

| Item | Priority | Estimated Effort |
|------|----------|-----------------|
| Missing API endpoints (evidence update, proposal lifecycle, task editing) | **High** | 1-2 days |
| Wire auth middleware to all routes | **High** | 2-3 hours |
| Citation verification on proposal rationale text | **Medium** | 4-6 hours |
| Prompt registry with versioned file loading | **Medium** | 4-6 hours |
| Expand eval golden set to 20-50 entries | **Medium** | 1 day |
| HDBSCAN + UMAP clustering upgrade | **Low** | 1-2 days |
| Multi-provider LLM abstraction (OpenAI/Claude support) | **Low** | 1 day |
| Celery + Redis migration (optional) | **Low** | 2-3 days |
| Metrics endpoint + structured logging | **Low** | 4-6 hours |
| Docker production config | **Low** | 4-6 hours |

---

## Task 1: Missing API Endpoints

### 1A. Evidence Update — `PUT /api/v1/evidence/{id}`

**File to edit:** `app/routers/evidence.py` + `app/services/evidence_service.py`

**Schema to add** in `app/schemas/evidence.py`:
```python
class EvidenceUpdate(BaseModel):
    title: str | None = None
    source_type: SourceType | None = None
    persona: str | None = None
    segment: str | None = None
    source_date: date | None = None
    metadata: dict | None = None
    # NOTE: raw_text is NOT updatable — changing text would invalidate all
    # chunks, problem mentions, embeddings, and clusters downstream.
    # If text changes, delete + re-upload.
```

**Service logic:**
- Fetch evidence by ID, return 404 if not found
- Apply only non-None fields (partial update)
- Commit and return updated evidence
- Do **NOT** allow `raw_text` updates — document why in a comment

**Endpoint:**
```python
@router.put("/evidence/{evidence_id}", response_model=EvidenceResponse)
async def update_evidence_endpoint(
    evidence_id: UUID,
    payload: EvidenceUpdate,
    session: AsyncSession = Depends(get_session),
) -> EvidenceResponse:
    ...
```

---

### 1B. Proposal Update — `PUT /api/v1/proposals/{id}`

**File to edit:** `app/routers/clusters.py` + `app/services/cluster_service.py` (or `proposal_service.py`)

**Schema to add** in `app/schemas/clusters.py`:
```python
class ProposalUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    impact: str | None = None    # high/medium/low
    effort: str | None = None    # S/M/L/XL
    metadata: dict | None = None
```

**Service logic:**
1. Fetch proposal by ID, 404 if not found
2. Increment `version` field by 1
3. Save a new `ProposalVersion` snapshot of the **current** state BEFORE applying changes (so history is preserved)
4. Apply the partial update fields
5. Commit and return

**Important:** The version snapshot must capture the state *before* the edit — this is how the versioning system tracks history per the strategy doc.

---

### 1C. Proposal Delete — `DELETE /api/v1/proposals/{id}`

**Simple cascade delete.** The ORM relationships already have `cascade="all, delete-orphan"` configured, so deleting a proposal will automatically remove:
- All `ProposalCitation` rows
- All `ProposalVersion` rows  
- All `Task` rows (via FK cascade)
- The `PriorityScore` row (via FK cascade)

```python
@router.delete("/proposals/{proposal_id}", status_code=204)
async def delete_proposal_endpoint(
    proposal_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    proposal = await session.get(FeatureProposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    await session.delete(proposal)
    await session.commit()
```

---

### 1D. Proposal Approve/Reject — `POST /api/v1/proposals/{id}/approve` and `/reject`

**Current state:** The `FeatureProposal` model does NOT have a `status` column. The strategy doc specifies `status VARCHAR(20) DEFAULT 'draft'` with values `draft`, `approved`, `rejected`, `archived`.

**Steps:**
1. **Add a new Alembic migration** (`005_add_proposal_status.py`):
   ```python
   op.add_column('feature_proposals', sa.Column('status', sa.Text(), server_default='draft', nullable=False))
   ```
2. **Update the ORM model** in `app/models/clusters.py` — add:
   ```python
   status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'draft'"))
   ```
3. **Add endpoints:**
   ```python
   @router.post("/proposals/{proposal_id}/approve")
   async def approve_proposal(proposal_id: UUID, ...) -> dict:
       proposal = await session.get(FeatureProposal, proposal_id)
       if not proposal:
           raise HTTPException(404)
       proposal.status = "approved"
       # Create a ProposalVersion snapshot
       version = ProposalVersion(
           proposal_id=proposal.id,
           version_number=proposal.version + 1,
           snapshot={...current state...},
           change_reason="Approved by PM",
       )
       proposal.version += 1
       session.add(version)
       await session.commit()
       return {"proposal_id": proposal_id, "status": "approved"}
   ```
4. Same pattern for `/reject` with `status = "rejected"`

**Schema updates:** Add `status` to `ProposalResponse` and `ProposalDetailResponse` in `app/schemas/clusters.py`.

---

### 1E. Proposal Regenerate — `POST /api/v1/proposals/{id}/regenerate`

**Logic:**
1. Load the proposal and its parent cluster
2. Save a `ProposalVersion` snapshot of current state (change_reason: "Before regeneration")
3. Call `generate_proposal_for_cluster()` from `proposal_service.py` — but instead of creating a new row, **update** the existing proposal in-place
4. Increment version, save new snapshot
5. Recalculate priority score

This requires refactoring `generate_proposal_for_cluster()` to accept an optional `existing_proposal_id` parameter. If provided, it updates in-place rather than creating a new row.

---

### 1F. Task Editing — `PATCH /api/v1/tasks/{id}`

**File to edit:** `app/routers/tasks.py`

**Schema to add** in `app/schemas/tasks.py`:
```python
class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    category: str | None = None  # backend|frontend|data|qa
    acceptance_criteria: list[str] | None = None
    estimated_effort: str | None = None  # XS|S|M|L|XL
    sort_order: int | None = None
```

**Endpoint:**
```python
@router.patch("/tasks/{task_id}", response_model=TaskResponse)
async def update_task_endpoint(
    task_id: UUID,
    payload: TaskUpdate,
    session: AsyncSession = Depends(get_session),
) -> TaskResponse:
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(task, field, value)
    await session.commit()
    return TaskResponse.model_validate(task)
```

---

### 1G. List All Proposals — `GET /api/v1/proposals`

Currently, proposals can only be viewed individually (`GET /proposals/{id}`) or via the roadmap. Add a paginated list endpoint:

```python
@router.get("/proposals")
async def list_proposals_endpoint(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: str | None = None,  # filter by draft/approved/rejected
    session: AsyncSession = Depends(get_session),
) -> dict:
    # Query FeatureProposal, join ProblemCluster for cluster_label
    # Filter by status if provided
    # Paginate, return items + total
    ...
```

---

## Task 2: Wire Authentication to All Routes

**Current state:** `app/middleware/auth.py` defines `RequireAuth = Annotated[str | None, Depends(verify_api_key)]` but no router imports or uses it. All endpoints are unauthenticated.

**Implementation approach (two options, pick one):**

### Option A: Per-Router Dependency (Recommended)

Add `RequireAuth` as a dependency to each router:

```python
# In each router file (evidence.py, problems.py, jobs.py, clusters.py, tasks.py):
from app.middleware.auth import RequireAuth

@router.post("/evidence", response_model=EvidenceResponse, status_code=201)
async def create_evidence_endpoint(
    payload: EvidenceCreate,
    _auth: RequireAuth,  # ← Add this parameter to every endpoint
    session: AsyncSession = Depends(get_session),
) -> EvidenceResponse:
    ...
```

### Option B: Global Router Dependencies

Apply auth at the router level so all endpoints in that router require it:

```python
# In main.py, when registering routers:
from app.middleware.auth import verify_api_key

auth_dep = [Depends(verify_api_key)]
app.include_router(evidence.router, prefix="/api/v1", tags=["evidence"], dependencies=auth_dep)
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"], dependencies=auth_dep)
# ... etc
```

**Exception:** Keep `GET /api/v1/health` unauthenticated (it's defined directly on the app, not on a router).

**Testing:** With `API_KEYS=""` in `.env`, auth is automatically disabled (dev mode). Set `API_KEYS=test-key-123` to test. Requests need `X-API-Key: test-key-123` header.

---

## Task 3: Citation Verification Pipeline

**Strategy reference:** Section "Reliability & Quality Guarantees" → "No Claim Without Citations"

**Current state:** When `proposal_service.py` generates a proposal, it auto-cites ALL cluster members as `ProposalCitation` rows. But it does NOT verify that `[Quote: "..."]` citations in the rationale text actually correspond to real quotes in the source data.

**What to build:** A post-processing step in `proposal_service.py` after LLM generation:

```python
# Add to app/services/proposal_service.py (or create app/utils/citations.py)

import re
from thefuzz import fuzz

CITATION_PATTERN = re.compile(r'\[Quote:\s*"([^"]+)"\]', re.IGNORECASE)

def verify_rationale_citations(
    rationale: str, 
    member_quotes: list[str],  # all quote_text values from cluster members
    threshold: float = 0.85,
) -> tuple[str, list[dict]]:
    """
    Verify [Quote: "..."] citations in rationale against actual member quotes.
    
    Returns:
        - cleaned_rationale: rationale with unverifiable citations removed
        - verified_citations: list of {quote_text, matched_source, score}
    """
    citations = CITATION_PATTERN.findall(rationale)
    verified = []
    cleaned = rationale
    
    for cited_text in citations:
        best_score = 0
        best_source = None
        for source_quote in member_quotes:
            score = fuzz.partial_ratio(cited_text.lower(), source_quote.lower()) / 100
            if score > best_score:
                best_score = score
                best_source = source_quote
        
        if best_score >= threshold:
            verified.append({
                "quote_text": cited_text,
                "matched_source": best_source,
                "score": best_score,
            })
        else:
            # Strip the unverifiable citation from rationale
            cleaned = cleaned.replace(f'[Quote: "{cited_text}"]', cited_text)
            logger.warning("Stripped unverifiable citation: %.50s... (best score: %.2f)", 
                          cited_text, best_score)
    
    return cleaned, verified
```

**Integration point:** Call this in `generate_proposal_for_cluster()` after step 4 (create proposal), before step 5 (auto-cite):

```python
# After generating proposal description/rationale from LLM:
member_quotes = [m.quote_text for m in members]
cleaned_description, verified = verify_rationale_citations(
    proposal.description, member_quotes, threshold=0.85
)
proposal.description = cleaned_description
# Use verified citations instead of blind auto-citation
```

---

## Task 4: Prompt Registry & Versioned Prompt Files

**Strategy reference:** Section "LLM Orchestration" → "Prompt Management"

**Current state:** Prompts are hardcoded inline strings in service files:
- `extraction_service.py` → `_build_prompt()`, version `"extract_problems_v1"`
- `cluster_service.py` → cluster summarization prompt, no version string
- `proposal_service.py` → `_build_proposal_prompt()`, version `"generate_proposal_v1"`
- `task_tree_service.py` → `_build_task_prompt()`, version `"generate_tasks_v1"`

**What to build:**

### Step 1: Create prompt files directory

```
backend/
├── prompts/
│   ├── extract_problems_v1.0.txt
│   ├── extract_problems_v1.1.txt    # (future iterations)
│   ├── summarize_cluster_v1.0.txt
│   ├── generate_proposal_v1.0.txt
│   └── generate_tasks_v1.0.txt
```

Each file is a Jinja2-style template or Python `.format()` template with named placeholders:

**Example `prompts/extract_problems_v1.0.txt`:**
```
You are extracting customer problems from a transcript chunk.
Return valid JSON only, with this schema:
{{
  "problems": [
    {{
      "problem_statement": "string",
      "severity": "critical|high|medium|low",
      "quote_text": "direct quote from the chunk",
      "persona": "optional",
      "segment": "optional",
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

If no problems are present, return {{"problems": []}}.

Chunk:
{chunk_text}
```

### Step 2: Create registry module

**File:** `app/llm/prompts.py`

```python
"""Prompt registry — loads and manages versioned prompt templates."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

PROMPT_REGISTRY: dict[str, dict[str, str]] = {}  # {name: {version: filepath}}
_LATEST: dict[str, str] = {}  # {name: latest_version}

def _discover_prompts() -> None:
    """Scan prompts/ directory and build registry."""
    if not PROMPTS_DIR.exists():
        logger.warning("Prompts directory not found: %s", PROMPTS_DIR)
        return
    for f in sorted(PROMPTS_DIR.glob("*.txt")):
        # Parse filename: extract_problems_v1.0.txt → name=extract_problems, version=v1.0
        stem = f.stem  # e.g., "extract_problems_v1.0"
        parts = stem.rsplit("_v", 1)
        if len(parts) != 2:
            logger.warning("Skipping unrecognized prompt file: %s", f.name)
            continue
        name = parts[0]
        version = f"v{parts[1]}"
        PROMPT_REGISTRY.setdefault(name, {})[version] = str(f)
        _LATEST[name] = version  # last sorted = latest version

_discover_prompts()

def get_prompt(name: str, version: str | None = None) -> str:
    """Load prompt template by name and optional version. Defaults to latest."""
    versions = PROMPT_REGISTRY.get(name)
    if not versions:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(PROMPT_REGISTRY.keys())}")
    target_version = version or _LATEST.get(name)
    filepath = versions.get(target_version)
    if not filepath:
        raise ValueError(f"Version {target_version} not found for prompt {name}. Available: {list(versions.keys())}")
    return Path(filepath).read_text(encoding="utf-8")

def get_prompt_version(name: str) -> str:
    """Get the latest version string for a prompt name."""
    return _LATEST.get(name, "unknown")

def list_prompts() -> dict[str, list[str]]:
    """Return all registered prompts and their versions."""
    return {name: sorted(versions.keys()) for name, versions in PROMPT_REGISTRY.items()}
```

### Step 3: Refactor services to use registry

In each service file, replace the inline prompt string with:
```python
from app.llm.prompts import get_prompt, get_prompt_version

template = get_prompt("extract_problems")
prompt = template.format(chunk_text=chunk.chunk_text)
version = get_prompt_version("extract_problems")
```

### Step 4: Add a prompt listing endpoint

In `app/routers/jobs.py`:
```python
@router.get("/prompts")
async def list_prompts_endpoint() -> dict:
    from app.llm.prompts import list_prompts
    return {"prompts": list_prompts()}
```

---

## Task 5: Expand Eval Harness Golden Set

**Current state:** `app/eval/golden_set.json` has 2 entries with 3 expected problems.  
**Target:** 20-50 entries covering diverse scenarios.

**What to create:** Add entries to `golden_set.json` covering:

### Categories to cover (aim for 3-5 entries per category):

1. **Single clear problem** — One obvious pain point with explicit quote
2. **Multiple problems in one chunk** — 2-4 distinct issues in the same paragraph
3. **No problems** — Positive feedback or neutral text → expected `"problems": []`
4. **Ambiguous severity** — Borderline cases to test severity calibration
5. **Implicit problems** — Pain expressed indirectly ("I wish..." / "It would be nice if...")
6. **Multi-persona** — Chunk mentions different user types experiencing different issues
7. **Technical jargon** — Domain-specific language that should still produce clean extractions
8. **Edge cases** — Very short chunks, single-sentence complaints, chunks with special characters

### Entry format:
```json
{
  "chunk_text": "...",
  "expected_problems": [
    {
      "problem_statement": "...",
      "severity": "critical|high|medium|low",
      "quote_text": "exact substring from chunk_text"
    }
  ],
  "notes": "optional - why this test case matters"
}
```

**Important:** `quote_text` must be an exact substring of `chunk_text` — the eval harness will verify this.

### Also add automated CI check:

Create `app/eval/test_golden_set_integrity.py`:
```python
"""Validate golden_set.json structural integrity (no LLM calls needed)."""
import json
from pathlib import Path

def test_golden_set_structure():
    golden = json.loads(Path("app/eval/golden_set.json").read_text())
    assert len(golden) >= 20, f"Golden set too small: {len(golden)} entries (need ≥20)"
    for i, entry in enumerate(golden):
        assert "chunk_text" in entry, f"Entry {i}: missing chunk_text"
        assert "expected_problems" in entry, f"Entry {i}: missing expected_problems"
        for j, prob in enumerate(entry["expected_problems"]):
            assert prob["quote_text"] in entry["chunk_text"], \
                f"Entry {i}, problem {j}: quote_text not found in chunk_text"
            assert prob["severity"] in ("critical", "high", "medium", "low"), \
                f"Entry {i}, problem {j}: invalid severity '{prob['severity']}'"
```

---

## Task 6: HDBSCAN + UMAP Clustering Upgrade

**Strategy reference:** Section "Core Services" → "D) Clustering Service" → "Phase 2 — HDBSCAN Upgrade"

**Current state:** `cluster_service.py` implements greedy threshold clustering with running-mean centroids. Works fine for small datasets but is order-dependent and doesn't discover density-based clusters.

**What to build:** An alternative clustering mode, triggered when count > threshold OR by explicit parameter.

### Step 1: Add dependencies to `requirements.txt`:
```
umap-learn>=0.5.5
hdbscan>=0.8.33
scikit-learn>=1.4.0
```

### Step 2: Add to `app/services/cluster_service.py`:

```python
import hdbscan
import umap
import numpy as np

async def run_hdbscan_clustering(
    session: AsyncSession,
    min_cluster_size: int = 3,
    umap_components: int = 15,
) -> list[ProblemCluster]:
    """Density-based clustering using UMAP dimensionality reduction + HDBSCAN."""
    
    # 1. Load all embeddings
    rows = (await session.execute(
        select(ProblemEmbedding).options(selectinload(ProblemEmbedding.problem))
    )).scalars().all()
    
    if len(rows) < min_cluster_size:
        raise ValueError(f"Need at least {min_cluster_size} embeddings, have {len(rows)}")
    
    embeddings = np.array([r.embedding for r in rows], dtype=np.float32)
    
    # 2. Dimensionality reduction
    reducer = umap.UMAP(
        n_components=min(umap_components, len(rows) - 1),
        metric='cosine',
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    
    # 3. HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
    )
    labels = clusterer.fit_predict(reduced)
    
    # 4. Clear previous clusters
    await session.execute(delete(ClusterMembership))
    await session.execute(delete(ProblemCluster).where(
        ~ProblemCluster.id.in_(
            select(FeatureProposal.cluster_id)  # preserve clusters with proposals
        )
    ))
    await session.flush()
    
    # 5. Create new clusters
    created = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise label
    
    for cluster_label in unique_labels:
        member_indices = [i for i, l in enumerate(labels) if l == cluster_label]
        member_embeddings = embeddings[member_indices]
        centroid = member_embeddings.mean(axis=0)
        
        first_problem = rows[member_indices[0]].problem
        cluster = ProblemCluster(
            label=first_problem.problem_statement[:120],
            centroid=centroid.tolist(),
            threshold=0.0,  # N/A for HDBSCAN
            mention_count=len(member_indices),
            metadata_={"algorithm": "hdbscan", "min_cluster_size": min_cluster_size},
        )
        session.add(cluster)
        await session.flush()
        
        for idx in member_indices:
            membership = ClusterMembership(
                cluster_id=cluster.id,
                problem_id=rows[idx].problem_id,
                similarity=float(np.dot(embeddings[idx], centroid) / 
                    (np.linalg.norm(embeddings[idx]) * np.linalg.norm(centroid) + 1e-9)),
            )
            session.add(membership)
        
        created.append(cluster)
    
    # 6. Handle noise points (label == -1) — log count
    noise_count = sum(1 for l in labels if l == -1)
    logger.info("HDBSCAN: %d clusters, %d noise points out of %d total",
                len(created), noise_count, len(rows))
    
    await session.commit()
    return created
```

### Step 3: Add endpoint

```python
@router.post("/clusters/run_hdbscan")
async def run_hdbscan_endpoint(
    min_cluster_size: int = Query(3, ge=2, le=50),
    session: AsyncSession = Depends(get_session),
) -> dict:
    clusters = await run_hdbscan_clustering(session, min_cluster_size=min_cluster_size)
    return {"clusters_created": len(clusters), "algorithm": "hdbscan"}
```

### Step 4: Auto-select algorithm

Optionally, update `POST /clusters/run` to auto-detect:
```python
@router.post("/clusters/run")
async def run_clustering_endpoint(
    threshold: float = Query(0.75, ge=0.0, le=1.0),
    algorithm: str = Query("auto"),  # "threshold", "hdbscan", "auto"
    session: AsyncSession = Depends(get_session),
) -> dict:
    if algorithm == "auto":
        count = (await session.execute(select(func.count(ProblemEmbedding.id)))).scalar() or 0
        algorithm = "hdbscan" if count > 500 else "threshold"
    
    if algorithm == "hdbscan":
        clusters = await run_hdbscan_clustering(session)
    else:
        clusters = await run_threshold_clustering(session, threshold=threshold)
    
    return {"clusters_created": len(clusters), "algorithm": algorithm}
```

---

## Task 7: Multi-Provider LLM Abstraction

**Strategy reference:** Section "Tech Stack" — "architecture supports swapping to OpenAI/Claude"

**Current state:** `app/llm/client.py` has a concrete `GeminiClient` class. No abstraction layer.

**What to build:**

### Step 1: Define abstract interface

```python
# app/llm/base.py

from abc import ABC, abstractmethod
from typing import Any

class LLMProvider(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def generate_json(self, prompt: str, prompt_version: str | None = None) -> dict[str, Any]:
        """Generate structured JSON from a prompt."""
        ...
    
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate a text embedding vector."""
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
```

### Step 2: Wrap existing GeminiClient

Refactor `GeminiClient` to extend `LLMProvider`. Keep all existing behavior.

### Step 3: Add OpenAI provider

```python
# app/llm/openai_client.py

import json
from openai import OpenAI
from app.llm.base import LLMProvider

class OpenAIClient(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self._model = model
    
    def generate_json(self, prompt: str, prompt_version: str | None = None) -> dict:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    
    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    
    @property
    def model_name(self) -> str:
        return self._model
```

### Step 4: Factory function in `client.py`

```python
def get_client() -> LLMProvider:
    global _client
    if _client is None:
        provider = getattr(settings, "llm_provider", "gemini")
        if provider == "openai":
            from app.llm.openai_client import OpenAIClient
            _client = OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_model)
        else:
            _client = GeminiClient()
    return _client
```

### Step 5: Add config

In `app/config.py`:
```python
llm_provider: str = "gemini"  # "gemini" | "openai"
openai_api_key: str = ""
openai_model: str = "gpt-4o"
```

**Note:** Embedding dimensions differ between providers! Gemini = 768, OpenAI `text-embedding-3-small` = 1536. If switching providers, you'll need to re-embed all problems. Add a check in the embedding service that warns if the vector dimension doesn't match pgvector column definition (768).

---

## Task 8: Celery + Redis Job Queue

**Strategy reference:** Section "Tech Stack" → "Job Queue: Celery + Redis"

**Current state:** Jobs use FastAPI `BackgroundTasks` with DB-persistent state via the `Job` model. Redis is provisioned in docker-compose but unused. This works but:
- Jobs are lost if the process crashes mid-execution
- No retry mechanism at the queue level
- No separate worker process (all work happens on the API server)
- No concurrency control beyond the in-code semaphore

**Priority: LOW.** The current BackgroundTasks approach is adequate for single-server deployments. Only migrate to Celery if you need:
- Horizontal scaling (multiple workers)
- Guaranteed delivery (job survives server restart)
- Priority queues

**If you proceed:**

### Step 1: Add `app/worker.py`
```python
from celery import Celery
from app.config import settings

celery_app = Celery("nexus", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,  # for reliability
    worker_prefetch_multiplier=1,
)
```

### Step 2: Convert service functions to Celery tasks
Each LLM job function (extract, embed, generate_proposal, generate_tasks) becomes a `@celery_app.task`.

### Step 3: Add worker service to `docker-compose.yml`
```yaml
worker:
  build: ./backend
  command: celery -A app.worker worker --loglevel=info --concurrency=4
  env_file: ./backend/.env
  environment:
    - DATABASE_URL=postgresql+asyncpg://nexus:nexus@db:5432/nexus
    - REDIS_URL=redis://redis:6379/0
  depends_on:
    - db
    - redis
```

### Step 4: Update `routers/jobs.py`
Replace `BackgroundTasks.add_task()` with `celery_task.delay()`. Keep the `Job` DB model for status tracking.

---

## Task 9: Observability & Metrics Endpoint

### 9A. Metrics Endpoint

Add `GET /api/v1/metrics` returning operational stats:

```python
@app.get("/api/v1/metrics")
async def metrics_endpoint(session: AsyncSession = Depends(get_session)) -> dict:
    from sqlalchemy import func, select
    from app.models.evidence import Evidence
    from app.models.problems import ProblemMention
    from app.models.clusters import ProblemCluster, FeatureProposal
    from app.models.jobs import Job, LLMCallLog
    
    evidence_count = (await session.execute(select(func.count(Evidence.id)))).scalar()
    problem_count = (await session.execute(select(func.count(ProblemMention.id)))).scalar()
    cluster_count = (await session.execute(select(func.count(ProblemCluster.id)))).scalar()
    proposal_count = (await session.execute(select(func.count(FeatureProposal.id)))).scalar()
    
    # Job stats
    job_stats = (await session.execute(
        select(Job.status, func.count(Job.id)).group_by(Job.status)
    )).all()
    
    # LLM cost totals from DB
    cost_totals = (await session.execute(
        select(
            func.count(LLMCallLog.id),
            func.coalesce(func.sum(LLMCallLog.cost_usd), 0),
            func.coalesce(func.sum(LLMCallLog.input_tokens), 0),
            func.coalesce(func.sum(LLMCallLog.output_tokens), 0),
        )
    )).one()
    
    return {
        "entities": {
            "evidence": evidence_count,
            "problems": problem_count,
            "clusters": cluster_count,
            "proposals": proposal_count,
        },
        "jobs": {status: count for status, count in job_stats},
        "llm": {
            "total_calls": cost_totals[0],
            "total_cost_usd": round(float(cost_totals[1]), 4),
            "total_input_tokens": cost_totals[2],
            "total_output_tokens": cost_totals[3],
        },
    }
```

### 9B. Structured Logging

Replace `logging.basicConfig` in `main.py` with structured JSON logging:

Add to `requirements.txt`:
```
structlog>=24.1.0
```

Create `app/logging_config.py`:
```python
import structlog
import logging

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()  # switch to JSONRenderer() in production
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
```

---

## Task 10: Docker Production Hardening

**Current state:** `Dockerfile.prod` exists but may not be complete. `docker-compose.yml` uses `--reload` for hot-reloading.

### Create `docker-compose.prod.yml`:

```yaml
services:
  api:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://nexus:${DB_PASSWORD}@db:5432/nexus
      - REDIS_URL=redis://redis:6379/0
      - API_KEYS=${API_KEYS}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      resources:
        limits:
          memory: 1G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: nexus
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: nexus
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U nexus -d nexus"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    # No port mapping (not exposed externally)

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    # No port mapping

volumes:
  pgdata:
```

### Update `Dockerfile.prod`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# No --reload, multiple workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Add Alembic migration step

Create an entrypoint script or docker-compose `command` override:
```bash
#!/bin/bash
alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Codebase Patterns & Conventions

Follow these existing patterns consistently:

### Router Pattern
```python
@router.method("/path", response_model=ResponseSchema, status_code=2xx)
async def endpoint_name(
    path_param: UUID,
    query_param: type = Query(default, ge=..., le=...),
    body_param: RequestSchema = ...,            # for POST/PUT/PATCH
    session: AsyncSession = Depends(get_session),
) -> ResponseSchema:
    try:
        result = await service_function(session, ...)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ResponseSchema.model_validate(result)
```

### Service Pattern
- Services take an `AsyncSession` as first parameter
- Raise `ValueError` for "not found" conditions (routers convert to 404)
- Use `selectinload` for eager loading relationships
- Call `session.flush()` mid-transaction for auto-generated UUIDs, `session.commit()` at the end

### Schema Pattern  
- `*Create` for POST bodies, `*Update` for PUT/PATCH bodies
- `*Response` for individual responses, `*ListResponse` for paginated lists
- All responses use `model_validate(orm_obj)` from `ConfigDict(from_attributes=True)`

### LLM Pattern
- Prompt builder function: `_build_*_prompt()` returns a string
- LLM call: `await call_with_retry(client.generate_json, prompt, PROMPT_VERSION, label="...")`
- The `call_with_retry` utility is in `app/utils/retry.py`

### Pagination Pattern
```python
items, total = await service_list_function(session, page=page, per_page=per_page)
return {
    "items": [...],
    "total": total,
    "page": page,
    "per_page": per_page,
    "total_pages": max(1, math.ceil(total / per_page)),
}
```

---

## File Reference Map

```
backend/
├── app/
│   ├── main.py                          # App entry, router registration, middleware
│   ├── config.py                        # Pydantic BaseSettings (.env loading)
│   ├── database.py                      # AsyncSession factory, engine config
│   ├── models/
│   │   ├── __init__.py                  # Re-exports all models
│   │   ├── evidence.py                  # Evidence, EvidenceChunk
│   │   ├── problems.py                  # ProblemMention
│   │   ├── embeddings.py                # ProblemEmbedding (pgvector)
│   │   ├── clusters.py                  # ProblemCluster, ClusterMembership, FeatureProposal, ProposalVersion, ProposalCitation
│   │   ├── tasks.py                     # Task (hierarchical)
│   │   ├── priority_scores.py           # PriorityScore
│   │   └── jobs.py                      # Job, LLMCallLog
│   ├── schemas/
│   │   ├── evidence.py                  # EvidenceCreate, EvidenceResponse, etc.
│   │   ├── problems.py                  # ProblemMentionCreate, LLMProblemsResponse, etc.
│   │   ├── clusters.py                  # ClusterResponse, ProposalCreate/Response, RoadmapItem
│   │   ├── tasks.py                     # TaskResponse, TaskTreeResponse
│   │   ├── priority_scores.py           # PriorityScoreResponse, StrategicWeightUpdate
│   │   ├── embeddings.py               # EmbedProblemsRequest
│   │   └── jobs.py                      # JobResponse, JobStatusResponse
│   ├── routers/
│   │   ├── evidence.py                  # POST/GET/DELETE evidence
│   │   ├── problems.py                  # GET problems, similar, stats
│   │   ├── jobs.py                      # POST extract/embed, GET status, GET llm/costs
│   │   ├── clusters.py                  # Clusters + proposals + roadmap + LLM generation
│   │   └── tasks.py                     # GET proposal tasks
│   ├── services/
│   │   ├── evidence_service.py          # Ingest + chunk + CRUD
│   │   ├── extraction_service.py        # LLM extraction + quote verification
│   │   ├── embeddings_service.py        # Vector embedding generation
│   │   ├── cluster_service.py           # Threshold clustering + summaries
│   │   ├── proposal_service.py          # LLM proposal generation
│   │   ├── task_tree_service.py         # LLM task tree generation
│   │   ├── prioritization_service.py    # Scoring formula
│   │   └── problems_service.py          # List, filter, similarity search, stats
│   ├── llm/
│   │   └── client.py                    # GeminiClient + LLMCallRecord + cost tracking
│   ├── middleware/
│   │   ├── auth.py                      # API key auth (NOT wired to routes)
│   │   └── rate_limit.py                # Sliding window rate limiter (ACTIVE)
│   ├── utils/
│   │   ├── chunking.py                  # Sentence-aware text chunking
│   │   └── retry.py                     # call_with_retry helper
│   └── eval/
│       ├── harness.py                   # Extraction quality eval runner
│       └── golden_set.json              # 2 test entries (needs expansion)
├── alembic/
│   └── versions/
│       ├── 001_phase1_tables.py         # evidence, chunks, problems, embeddings
│       ├── 002_phase2_clusters.py       # clusters, memberships, proposals, citations
│       ├── 003_phase3_jobs_llm.py       # jobs, llm_call_log
│       └── 004_phase4_tasks_priorities.py # tasks, priority_scores, proposal_versions
├── prompts/                             # TO CREATE — versioned prompt templates
├── docker-compose.yml                   # Dev compose (root of workspace)
├── Dockerfile                           # Dev container
├── Dockerfile.prod                      # Production container
├── requirements.txt                     # Python dependencies
└── strategybackend.md                   # Full strategy document (READ THIS)
```

---

## Recommended Implementation Order

1. **Task 1** (Missing endpoints) — Highest impact, enables frontend integration
2. **Task 2** (Auth wiring) — Security baseline, quick win
3. **Task 3** (Citation verification) — Core quality guarantee from strategy
4. **Task 4** (Prompt registry) — Enables A/B testing and iteration
5. **Task 5** (Golden set expansion) — Confidence in extraction quality
6. **Task 9** (Metrics endpoint) — Operational visibility
7. **Task 6** (HDBSCAN) — Only if dataset grows past ~500 mentions
8. **Task 7** (Multi-provider) — Only if Gemini limitations encountered
9. **Task 10** (Docker prod) — Before any public deployment
10. **Task 8** (Celery) — Only if scaling beyond single server

---

**Reference:** Read `strategybackend.md` for full context on architectural decisions, prompt designs, and quality guarantees. Read `ARCHITECTURE.md` for detailed documentation of everything already built.

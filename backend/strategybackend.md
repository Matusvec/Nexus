# Nexus Backend Strategy

> **"Cursor for Product Managers"** — Transform messy customer evidence into roadmap-grade decisions + dev-ready breakdowns, all traceable to quotes.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Data Model](#data-model)
4. [Core Services](#core-services)
5. [API Design](#api-design)
6. [LLM Orchestration](#llm-orchestration)
7. [Reliability & Quality Guarantees](#reliability--quality-guarantees)
8. [Development Phases](#development-phases)
9. [Deployment & Infrastructure](#deployment--infrastructure)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        API GATEWAY (FastAPI)                      │
│  Auth · Rate Limiting · Request Validation · CORS                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────────────────────┐
       │               │                               │
       ▼               ▼                               ▼
┌─────────────┐ ┌──────────────┐              ┌──────────────────┐
│  Evidence    │ │  Extraction  │              │  Prioritization  │
│  Service     │ │  Service     │              │  Engine          │
│              │ │  (async)     │              │                  │
│ - Ingest     │ │ - Problems   │              │ - Scoring model  │
│ - Chunk      │ │ - Structured │              │ - Roadmap rank   │
│ - Store      │ │   output     │              │ - Explainability │
└──────┬──────┘ └──────┬───────┘              └────────┬─────────┘
       │               │                               │
       ▼               ▼                               │
┌──────────────────────────────────┐                   │
│       Postgres + pgvector        │◄──────────────────┘
│  - Structured data (all tables)  │
│  - Vector embeddings             │
│  - Provenance chains             │
└──────────────┬───────────────────┘
               │
       ┌───────┼───────────────────┐
       │       │                   │
       ▼       ▼                   ▼
┌──────────┐ ┌──────────────┐ ┌──────────────────┐
│ Embedding│ │  Clustering  │ │  Proposal +      │
│ & Search │ │  Service     │ │  Task Tree Gen   │
│ Service  │ │              │ │                  │
│          │ │ - Threshold  │ │ - Feature specs  │
│ - Embed  │ │ - HDBSCAN    │ │ - Task breakdown │
│ - kNN    │ │ - Summarize  │ │ - Acceptance     │
└──────────┘ └──────────────┘ │   criteria       │
                              └──────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────┐
│  Job Queue   │ │  LLM     │ │  Observability   │
│  (Celery +   │ │  Provider│ │                  │
│   Redis)     │ │  (Gemini/│ │ - Structured logs│
│              │ │  OpenAI) │ │ - Prompt versions│
│ - Async jobs │ │          │ │ - Cost tracking  │
│ - Retries    │ │          │ │ - Eval harness   │
│ - Progress   │ │          │ │                  │
└──────────────┘ └──────────┘ └──────────────────┘
```

### Data Flow: End to End

```
Evidence Upload → Chunking → Problem Extraction (LLM) → Embedding
     │                                │
     ▼                                ▼
  Raw storage              Problem Mentions (structured)
                                      │
                                      ▼
                              Similarity Clustering
                                      │
                                      ▼
                              Pain Clusters (labeled)
                                      │
                                      ▼
                              Feature Proposals (LLM)
                                      │
                                      ▼
                              Task Trees (LLM)
                                      │
                                      ▼
                              Prioritized Roadmap (scoring)
```

---

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **API Framework** | FastAPI (Python) | Async-native, Pydantic validation, auto OpenAPI docs |
| **Database** | PostgreSQL + pgvector | Structured data + vector similarity in one DB, proven at scale |
| **Job Queue** | Celery + Redis | Async LLM jobs, retries, progress tracking; swap to lightweight queue if overengineered early |
| **LLM Provider** | Gemini 2.0 Flash (primary) | Fast, affordable, tool calling; architecture supports swapping to OpenAI/Claude |
| **Embeddings** | Gemini text-embedding-004 | Consistent with LLM provider; pgvector for storage + search |
| **Clustering** | Threshold clustering → HDBSCAN | Start simple, upgrade when data volume justifies |
| **Dimensionality Reduction** | UMAP | Pre-clustering step for HDBSCAN phase |
| **Auth** | API key (v1) → JWT/OAuth (v2) | Simple early, secure later |
| **Deployment** | Docker Compose (v1) → Kubernetes (v3) | Local-first matches Nexus philosophy |
| **Observability** | Structured JSON logs + custom cost tracker | Essential for LLM cost control and prompt quality |

### Why Postgres + pgvector over ChromaDB?

The existing Nexus prototype uses ChromaDB for document RAG. The Product Manager pipeline requires:

- **Relational joins** (evidence → problems → clusters → proposals → tasks)
- **Transactional integrity** (multi-table writes during extraction)
- **Complex queries** (filter by persona × severity × date × tag)
- **Provenance chains** (every claim → citation → quote → source)

pgvector handles embeddings while Postgres handles everything else — one database, one backup, one migration tool. ChromaDB remains available for the document RAG features described in the original README.

---

## Data Model

### Entity Relationship Diagram

```
evidence (1) ──────< evidence_chunks (N)
                          │
                          │ (LLM extracts from chunks)
                          ▼
                    problem_mentions (N)
                          │
                          ├──── problem_embeddings (1:1)
                          │
                          ├────> cluster_members (N) >──── clusters (1)
                          │                                    │
                          │                                    │ (LLM generates)
                          │                                    ▼
                          │                            feature_proposals (N)
                          │                                    │
                          │                                    ├──── proposal_versions (N)
                          │                                    │
                          │                                    ├──── feature_citations (N) ───> back to problem_mentions
                          │                                    │
                          │                                    ├──── priority_scores (1:1)
                          │                                    │
                          │                                    └────> tasks (N, hierarchical)
                          │
                          └──── (quote_text links back to evidence_chunks)
```

### Table Definitions

#### `evidence`
```sql
CREATE TABLE evidence (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    source_type     VARCHAR(50) NOT NULL,  -- 'interview', 'support_ticket', 'sales_note', 'survey', 'other'
    persona         VARCHAR(100),
    segment         VARCHAR(100),
    source_date     DATE,
    raw_text        TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_evidence_source_type ON evidence(source_type);
CREATE INDEX idx_evidence_persona ON evidence(persona);
CREATE INDEX idx_evidence_segment ON evidence(segment);
CREATE INDEX idx_evidence_created_at ON evidence(created_at);
```

#### `evidence_chunks`
```sql
CREATE TABLE evidence_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_id     UUID NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    start_offset    INTEGER NOT NULL,  -- character offset in raw_text
    end_offset      INTEGER NOT NULL,
    token_count     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chunks_evidence_id ON evidence_chunks(evidence_id);
```

#### `problem_mentions`
```sql
CREATE TABLE problem_mentions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_id     UUID NOT NULL REFERENCES evidence(id),
    chunk_id        UUID NOT NULL REFERENCES evidence_chunks(id),
    problem_statement TEXT NOT NULL,           -- normalized problem description
    persona         VARCHAR(100),
    segment         VARCHAR(100),
    severity        VARCHAR(20) NOT NULL,      -- 'critical', 'high', 'medium', 'low'
    quote_text      TEXT NOT NULL,             -- direct quote from source
    quote_start     INTEGER,                   -- offset in chunk
    quote_end       INTEGER,
    tags            TEXT[] DEFAULT '{}',        -- ['pricing', 'onboarding', 'reliability']
    extraction_job_id UUID,
    prompt_version  VARCHAR(50),               -- tracks which prompt produced this
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_problems_evidence_id ON problem_mentions(evidence_id);
CREATE INDEX idx_problems_severity ON problem_mentions(severity);
CREATE INDEX idx_problems_tags ON problem_mentions USING GIN(tags);
CREATE INDEX idx_problems_persona ON problem_mentions(persona);
```

#### `problem_embeddings`
```sql
CREATE TABLE problem_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id      UUID NOT NULL UNIQUE REFERENCES problem_mentions(id) ON DELETE CASCADE,
    embedding       vector(768) NOT NULL,      -- Gemini embedding dimension
    model_version   VARCHAR(50) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_embeddings_vector ON problem_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

#### `clusters`
```sql
CREATE TABLE clusters (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label           TEXT NOT NULL,             -- short cluster name, e.g. "Onboarding is confusing"
    summary         TEXT,                      -- AI-generated summary
    member_count    INTEGER DEFAULT 0,
    avg_severity    FLOAT,
    top_quotes      JSONB DEFAULT '[]',        -- best supporting quotes
    algorithm       VARCHAR(50) DEFAULT 'threshold',  -- 'threshold', 'hdbscan'
    prompt_version  VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### `cluster_members`
```sql
CREATE TABLE cluster_members (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id      UUID NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    problem_id      UUID NOT NULL REFERENCES problem_mentions(id) ON DELETE CASCADE,
    similarity_score FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(cluster_id, problem_id)
);

CREATE INDEX idx_cluster_members_cluster ON cluster_members(cluster_id);
CREATE INDEX idx_cluster_members_problem ON cluster_members(problem_id);
```

#### `feature_proposals`
```sql
CREATE TABLE feature_proposals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id      UUID NOT NULL REFERENCES clusters(id),
    feature_name    TEXT NOT NULL,
    one_liner       TEXT NOT NULL,
    user_story      TEXT,                      -- "As a [persona], I want..."
    jtbd_framing    TEXT,                      -- Jobs to be done
    rationale       TEXT NOT NULL,             -- why this matters (with citations)
    success_metrics JSONB DEFAULT '[]',        -- [{metric, target, reasoning}]
    risks           JSONB DEFAULT '[]',        -- [{risk, mitigation, severity}]
    edge_cases      JSONB DEFAULT '[]',
    scope_estimate  VARCHAR(5) NOT NULL,       -- 'S', 'M', 'L', 'XL'
    status          VARCHAR(20) DEFAULT 'draft', -- 'draft', 'approved', 'rejected', 'archived'
    prompt_version  VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_proposals_cluster ON feature_proposals(cluster_id);
CREATE INDEX idx_proposals_status ON feature_proposals(status);
```

#### `feature_citations`
```sql
CREATE TABLE feature_citations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id     UUID NOT NULL REFERENCES feature_proposals(id) ON DELETE CASCADE,
    problem_id      UUID NOT NULL REFERENCES problem_mentions(id),
    citation_context TEXT NOT NULL,            -- which claim this citation supports
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_citations_proposal ON feature_citations(proposal_id);
```

#### `proposal_versions`
```sql
CREATE TABLE proposal_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id     UUID NOT NULL REFERENCES feature_proposals(id) ON DELETE CASCADE,
    version_number  INTEGER NOT NULL,
    snapshot        JSONB NOT NULL,            -- full proposal state at this version
    change_reason   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_versions_proposal ON proposal_versions(proposal_id);
```

#### `tasks`
```sql
CREATE TABLE tasks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id     UUID NOT NULL REFERENCES feature_proposals(id) ON DELETE CASCADE,
    parent_task_id  UUID REFERENCES tasks(id),  -- hierarchical nesting
    title           TEXT NOT NULL,
    description     TEXT,
    category        VARCHAR(20) NOT NULL,       -- 'backend', 'frontend', 'data', 'qa'
    acceptance_criteria JSONB DEFAULT '[]',      -- ["Given X, when Y, then Z"]
    estimated_effort VARCHAR(10),               -- 'XS', 'S', 'M', 'L', 'XL'
    dependencies    UUID[] DEFAULT '{}',         -- other task IDs
    sort_order      INTEGER DEFAULT 0,
    prompt_version  VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tasks_proposal ON tasks(proposal_id);
CREATE INDEX idx_tasks_parent ON tasks(parent_task_id);
CREATE INDEX idx_tasks_category ON tasks(category);
```

#### `priority_scores`
```sql
CREATE TABLE priority_scores (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id     UUID NOT NULL UNIQUE REFERENCES feature_proposals(id) ON DELETE CASCADE,
    frequency_score FLOAT NOT NULL,            -- how often this problem appears
    severity_score  FLOAT NOT NULL,            -- avg severity of cluster
    strategic_weight FLOAT DEFAULT 1.0,        -- manual adjustment factor
    effort_estimate FLOAT NOT NULL,            -- derived from scope_estimate
    final_score     FLOAT NOT NULL,            -- (freq × severity × weight) / effort
    score_breakdown JSONB NOT NULL,            -- full calculation for explainability
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_priority_score ON priority_scores(final_score DESC);
```

#### `jobs` (async job tracking)
```sql
CREATE TABLE jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type        VARCHAR(50) NOT NULL,      -- 'extract_problems', 'cluster', 'generate_proposal', 'generate_tasks'
    status          VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'running', 'completed', 'failed'
    input_params    JSONB NOT NULL,
    result          JSONB,
    error_message   TEXT,
    token_usage     JSONB,                     -- {prompt_tokens, completion_tokens, total_cost}
    prompt_version  VARCHAR(50),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type ON jobs(job_type);
```

---

## Core Services

### A) Evidence Service

**Responsibility:** Ingest, chunk, and store raw customer evidence.

**Chunking Strategy:**
- Token-aware splitting (max 500 tokens per chunk for extraction accuracy)
- Sentence-boundary aware (never split mid-sentence)
- Preserve character offsets for provenance tracing
- Overlap of ~50 tokens between chunks to avoid losing context at boundaries

**Key Logic:**
```python
# Pseudocode for evidence ingestion
async def ingest_evidence(payload: EvidenceCreate) -> Evidence:
    # 1. Store raw evidence
    evidence = await db.insert("evidence", payload)
    
    # 2. Chunk with offset tracking
    chunks = chunk_text(
        text=payload.raw_text,
        max_tokens=500,
        overlap_tokens=50,
        preserve_sentences=True
    )
    
    # 3. Store chunks with provenance
    for i, chunk in enumerate(chunks):
        await db.insert("evidence_chunks", {
            "evidence_id": evidence.id,
            "chunk_index": i,
            "chunk_text": chunk.text,
            "start_offset": chunk.start,
            "end_offset": chunk.end,
            "token_count": chunk.token_count
        })
    
    # 4. Enqueue extraction job
    job = await enqueue_job("extract_problems", {"evidence_id": evidence.id})
    
    return evidence
```

---

### B) Extraction Service

**Responsibility:** Convert evidence chunks into structured problem mentions via LLM.

**Critical Design Decisions:**
- **Strict JSON schema enforcement** — LLM output must conform to Pydantic model
- **Retry on malformed output** — up to 3 retries with increasingly explicit instructions
- **Quote verification** — extracted quote must exist in source chunk (fuzzy match with >90% similarity)
- **Batch processing** — process chunks in parallel within a single evidence document

**Extraction Prompt (versioned):**
```
PROMPT_VERSION: "extract_problems_v1.2"

You are analyzing a customer evidence document. Extract ALL distinct problems 
the customer mentions. For each problem:

1. problem_statement: A normalized, clear description of the problem (imperative form)
2. persona: Who is experiencing this problem (if identifiable)
3. severity: critical | high | medium | low (based on language intensity + impact)
4. quote_text: The EXACT words from the text that describe this problem (verbatim)
5. tags: Categorize with 1-3 tags from: [pricing, onboarding, reliability, 
   performance, ux, integration, security, reporting, permissions, billing, 
   documentation, mobile, collaboration]

Rules:
- quote_text MUST be an exact substring of the source text
- One problem per mention (don't merge distinct issues)
- Severity guide: critical = blocking/churn risk, high = significant pain, 
  medium = inconvenience, low = nice-to-have
- If no problems are found, return an empty array

Output JSON array of objects matching the schema exactly.
```

**Schema Validation:**
```python
class ProblemMention(BaseModel):
    problem_statement: str = Field(..., min_length=10, max_length=500)
    persona: Optional[str] = None
    severity: Literal["critical", "high", "medium", "low"]
    quote_text: str = Field(..., min_length=5)
    tags: List[str] = Field(default_factory=list, max_length=5)

class ExtractionResult(BaseModel):
    problems: List[ProblemMention]
```

---

### C) Embedding & Retrieval Service

**Responsibility:** Embed problem mentions and enable similarity search.

**Implementation:**
- Embed `problem_statement` field (not raw quote — normalized text clusters better)
- Use Gemini `text-embedding-004` (768 dimensions)
- Store in pgvector column
- kNN search via `<=>` cosine distance operator

**Key Operations:**
```python
async def embed_problem(problem_id: UUID, problem_statement: str):
    embedding = await gemini.embed(problem_statement)
    await db.insert("problem_embeddings", {
        "problem_id": problem_id,
        "embedding": embedding,
        "model_version": "text-embedding-004"
    })

async def find_similar(text: str, limit: int = 20, threshold: float = 0.8):
    query_embedding = await gemini.embed(text)
    return await db.query("""
        SELECT pm.*, pe.embedding <=> $1 AS distance
        FROM problem_embeddings pe
        JOIN problem_mentions pm ON pm.id = pe.problem_id
        WHERE pe.embedding <=> $1 < $2
        ORDER BY distance ASC
        LIMIT $3
    """, query_embedding, 1 - threshold, limit)
```

---

### D) Clustering Service

**Responsibility:** Group similar problem mentions into pain clusters.

**Phase 1 — Threshold Clustering (MVP):**
```python
async def cluster_problems(similarity_threshold: float = 0.82):
    # 1. Fetch all unassigned problem embeddings
    problems = await get_unclustered_problems()
    
    # 2. Greedy agglomerative clustering
    clusters = []
    for problem in problems:
        best_cluster = None
        best_similarity = 0
        
        for cluster in clusters:
            sim = cosine_similarity(problem.embedding, cluster.centroid)
            if sim > best_similarity and sim >= similarity_threshold:
                best_cluster = cluster
                best_similarity = sim
        
        if best_cluster:
            best_cluster.add(problem)
        else:
            clusters.append(new_cluster(problem))
    
    # 3. Generate labels + summaries for each cluster
    for cluster in clusters:
        label, summary, top_quotes = await summarize_cluster(cluster)
        await save_cluster(label, summary, top_quotes, cluster.members)
```

**Phase 2 — HDBSCAN Upgrade (when >500 mentions):**
```python
async def cluster_with_hdbscan():
    embeddings = await get_all_embeddings()  # (N, 768)
    
    # Dimensionality reduction
    reduced = umap.UMAP(n_components=15, metric='cosine').fit_transform(embeddings)
    
    # Density-based clustering
    labels = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean').fit_predict(reduced)
    
    # Generate cluster summaries
    for cluster_id in set(labels):
        if cluster_id == -1:  # noise
            continue
        members = [problems[i] for i, l in enumerate(labels) if l == cluster_id]
        await summarize_and_save_cluster(members)
```

**Cluster Summarization Prompt:**
```
Given these customer problem mentions:
{formatted_mentions}

Generate:
1. label: A short (3-8 word) label for this pain cluster
2. summary: A 2-3 sentence summary of the core issue
3. top_quotes: The 3 most compelling direct quotes from the mentions

The label should be actionable (e.g., "Onboarding flow is confusing" not "Onboarding issues").
```

---

### E) Proposal Generator Service

**Responsibility:** Convert pain clusters into structured feature proposals.

**Generation Prompt:**
```
PROMPT_VERSION: "generate_proposal_v1.0"

You are a senior product manager. Based on the following customer pain cluster,
generate a structured feature proposal.

CLUSTER:
- Label: {cluster.label}
- Summary: {cluster.summary}
- Member count: {cluster.member_count}
- Severity distribution: {severity_breakdown}
- Top quotes: {cluster.top_quotes}
- All problem mentions: {formatted_mentions}

Generate a feature proposal with these fields:
1. feature_name: Clear, concise name
2. one_liner: Single sentence describing the feature
3. user_story: "As a [persona], I want [goal] so that [benefit]"
4. jtbd_framing: Jobs-to-be-done framing of the need
5. rationale: Why this matters — MUST cite specific quotes using [Quote: "..."]
6. success_metrics: Array of {metric, target, reasoning}
7. risks: Array of {risk, mitigation, severity}
8. edge_cases: Array of potential edge cases
9. scope_estimate: S (< 1 week), M (1-3 weeks), L (1-2 months), XL (2+ months)

CRITICAL: Every claim in the rationale MUST include a [Quote: "exact quote"] citation.
Do not invent or hallucinate user quotes.
```

**Post-processing:**
- Verify all citations exist in cluster's problem mentions (fuzzy match)
- Strip any citation that can't be traced to source
- Store citation links in `feature_citations` table
- Create initial version in `proposal_versions`

---

### F) Task Tree Generator

**Responsibility:** Transform feature proposals into implementation-ready task trees.

**Generation Prompt:**
```
PROMPT_VERSION: "generate_tasks_v1.0"

You are a senior tech lead. Convert this feature proposal into an implementation
task tree.

PROPOSAL:
{feature_proposal_json}

Generate a hierarchical task tree with these categories:
- backend: API endpoints, services, data model changes
- frontend: Components, pages, state management, UI flows
- data: Migrations, indexes, data transformations
- qa: Test cases, acceptance criteria, edge case tests

For each task:
1. title: Clear, actionable task name
2. description: What needs to be built (2-3 sentences)
3. category: backend | frontend | data | qa
4. acceptance_criteria: Array of "Given X, when Y, then Z" statements
5. estimated_effort: XS (<2hrs), S (<1 day), M (1-3 days), L (3-5 days), XL (5+ days)
6. dependencies: Which other tasks must complete first (by title reference)

Rules:
- Tasks should be small enough for one developer to complete
- Every task must have at least one acceptance criterion
- Data migrations come before backend tasks that depend on them
- QA tasks reference the features they validate
```

**Task Tree Structure:**
```json
{
  "backend": [
    {
      "title": "Create evidence upload endpoint",
      "description": "...",
      "acceptance_criteria": ["Given a valid transcript, when POST /evidence, then return 201 with evidence ID"],
      "estimated_effort": "M",
      "subtasks": [
        { "title": "Add request validation schema", "estimated_effort": "S" },
        { "title": "Implement chunking logic", "estimated_effort": "M" }
      ]
    }
  ],
  "frontend": [...],
  "data": [...],
  "qa": [...]
}
```

---

### G) Prioritization Engine

**Responsibility:** Rank feature proposals with transparent, explainable scoring.

**Scoring Formula:**
```
final_score = (frequency_score × severity_score × strategic_weight) / effort_estimate
```

**Component Calculation:**
```python
def calculate_priority(proposal: FeatureProposal, cluster: Cluster) -> PriorityScore:
    # Frequency: how often this problem appears relative to total mentions
    total_mentions = await get_total_mention_count()
    frequency_score = cluster.member_count / total_mentions * 100  # normalized 0-100
    
    # Severity: weighted average of cluster member severities
    severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    severity_score = weighted_avg(cluster.members, severity_weights)  # 1-4 scale
    
    # Strategic weight: manual adjustment (default 1.0, PM can override)
    strategic_weight = proposal.strategic_weight or 1.0
    
    # Effort: derived from scope estimate
    effort_map = {"S": 1, "M": 3, "L": 8, "XL": 20}
    effort_estimate = effort_map[proposal.scope_estimate]
    
    final_score = (frequency_score * severity_score * strategic_weight) / effort_estimate
    
    return PriorityScore(
        frequency_score=frequency_score,
        severity_score=severity_score,
        strategic_weight=strategic_weight,
        effort_estimate=effort_estimate,
        final_score=final_score,
        score_breakdown={
            "formula": "(frequency × severity × weight) / effort",
            "frequency": {"value": frequency_score, "explanation": f"{cluster.member_count} mentions out of {total_mentions}"},
            "severity": {"value": severity_score, "distribution": severity_distribution},
            "weight": {"value": strategic_weight, "reason": "default" },
            "effort": {"value": effort_estimate, "scope": proposal.scope_estimate},
            "final": final_score
        }
    )
```

---

## API Design

### Complete Endpoint Map

```
Evidence
  POST   /api/v1/evidence                      → Upload new evidence
  GET    /api/v1/evidence                       → List all evidence (paginated)
  GET    /api/v1/evidence/{id}                  → Get evidence detail + chunks
  DELETE /api/v1/evidence/{id}                  → Delete evidence + cascade

Problems
  GET    /api/v1/problems                       → List all problem mentions (filterable)
  GET    /api/v1/problems/{id}                  → Get problem detail with source
  GET    /api/v1/problems/similar?text=...      → Similarity search
  GET    /api/v1/problems/stats                 → Aggregate stats (by persona, severity, tag)

Jobs
  POST   /api/v1/jobs/extract_problems          → Trigger extraction for evidence
  POST   /api/v1/jobs/cluster                   → Trigger re-clustering
  POST   /api/v1/jobs/generate_proposal         → Generate proposal from cluster
  POST   /api/v1/jobs/generate_tasks            → Generate tasks from proposal
  GET    /api/v1/jobs/{id}                      → Get job status + result

Clusters
  GET    /api/v1/clusters                       → List all clusters
  GET    /api/v1/clusters/{id}                  → Cluster detail + members + quotes

Feature Proposals
  GET    /api/v1/feature_proposals              → List proposals (filterable by status)
  GET    /api/v1/feature_proposals/{id}         → Proposal detail + citations
  PATCH  /api/v1/feature_proposals/{id}         → Edit proposal (light edits)
  POST   /api/v1/feature_proposals/{id}/approve → Approve proposal
  POST   /api/v1/feature_proposals/{id}/reject  → Reject proposal
  POST   /api/v1/feature_proposals/{id}/regenerate → Re-generate with updated prompt

Tasks
  GET    /api/v1/feature_proposals/{id}/tasks   → Get task tree for proposal
  PATCH  /api/v1/tasks/{id}                     → Edit individual task

Roadmap
  GET    /api/v1/roadmap                        → Ranked proposals with score breakdown
  PATCH  /api/v1/roadmap/{proposal_id}/weight   → Adjust strategic weight

Health
  GET    /api/v1/health                         → Service health check
  GET    /api/v1/metrics                        → Cost + usage metrics
```

### Request/Response Examples

**POST /api/v1/evidence**
```json
// Request
{
  "title": "Customer Interview - Acme Corp PM",
  "source_type": "interview",
  "persona": "Product Manager",
  "segment": "Enterprise",
  "source_date": "2026-01-15",
  "raw_text": "...the onboarding was really confusing. I spent 3 hours trying to set up my first project and still couldn't figure out permissions..."
}

// Response (201)
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Customer Interview - Acme Corp PM",
  "source_type": "interview",
  "chunk_count": 12,
  "extraction_job_id": "660e8400-e29b-41d4-a716-446655440001",
  "created_at": "2026-02-13T10:30:00Z"
}
```

**GET /api/v1/clusters/{id}**
```json
{
  "id": "770e8400-...",
  "label": "Onboarding flow is confusing",
  "summary": "Multiple users across enterprise and mid-market segments report difficulty completing initial setup. Key friction points include permissions configuration and project creation workflows.",
  "member_count": 23,
  "avg_severity": 3.2,
  "severity_distribution": { "critical": 4, "high": 11, "medium": 7, "low": 1 },
  "top_quotes": [
    { "text": "I spent 3 hours trying to set up my first project", "source": "Acme Corp PM Interview", "severity": "high" },
    { "text": "our team gave up on onboarding after day two", "source": "Support Ticket #4521", "severity": "critical" }
  ],
  "members": [ /* truncated problem_mention objects */ ]
}
```

**GET /api/v1/roadmap**
```json
{
  "proposals": [
    {
      "rank": 1,
      "proposal_id": "880e8400-...",
      "feature_name": "Guided Onboarding Wizard",
      "one_liner": "Step-by-step setup flow replacing the current blank-slate experience",
      "score": 42.5,
      "score_breakdown": {
        "frequency": { "value": 34.0, "explanation": "23 of 68 total mentions" },
        "severity": { "value": 3.2, "distribution": { "critical": 4, "high": 11 } },
        "weight": { "value": 1.2, "reason": "strategic priority: activation" },
        "effort": { "value": 3, "scope": "M" }
      },
      "cluster_label": "Onboarding flow is confusing",
      "status": "approved"
    }
  ],
  "total_proposals": 12,
  "last_clustered_at": "2026-02-12T18:00:00Z"
}
```

---

## LLM Orchestration

### Prompt Management

```python
# prompts/registry.py

PROMPT_REGISTRY = {
    "extract_problems": {
        "v1.0": "prompts/extract_problems_v1.0.txt",
        "v1.1": "prompts/extract_problems_v1.1.txt",  # improved severity calibration
        "v1.2": "prompts/extract_problems_v1.2.txt",  # added tag taxonomy
    },
    "summarize_cluster": {
        "v1.0": "prompts/summarize_cluster_v1.0.txt",
    },
    "generate_proposal": {
        "v1.0": "prompts/generate_proposal_v1.0.txt",
    },
    "generate_tasks": {
        "v1.0": "prompts/generate_tasks_v1.0.txt",
    }
}

# Every LLM call logs: prompt_version, input tokens, output tokens, cost, latency
```

### Error Handling & Retries

```python
async def call_llm_with_validation(
    prompt: str,
    response_schema: type[BaseModel],
    max_retries: int = 3
) -> BaseModel:
    for attempt in range(max_retries):
        try:
            raw_response = await llm.generate(prompt)
            parsed = response_schema.model_validate_json(raw_response)
            return parsed
        except ValidationError as e:
            if attempt < max_retries - 1:
                # Append error feedback to prompt
                prompt += f"\n\nYour previous response had errors: {e}\nPlease fix and try again."
            else:
                raise ExtractionError(f"Failed after {max_retries} attempts: {e}")
        except RateLimitError:
            await asyncio.sleep(2 ** attempt)  # exponential backoff
```

### Cost Tracking

```python
@dataclass
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    model: str
    cost_usd: float  # calculated from token counts + model pricing

    @staticmethod
    def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
        pricing = {
            "gemini-2.0-flash": {"input": 0.00001, "output": 0.00004},  # per token
            "gpt-4o": {"input": 0.000005, "output": 0.000015},
        }
        rates = pricing[model]
        return prompt_tokens * rates["input"] + completion_tokens * rates["output"]
```

---

## Reliability & Quality Guarantees

### 1. No Claim Without Citations

Every feature proposal's rationale is post-processed:
```python
async def verify_citations(proposal: FeatureProposal, cluster: Cluster):
    """Ensure every [Quote: "..."] in rationale exists in source data."""
    citations = extract_citation_patterns(proposal.rationale)
    verified = []
    for citation in citations:
        match = fuzzy_find_quote(citation.text, cluster.all_quotes, threshold=0.90)
        if match:
            verified.append(CitationLink(
                proposal_id=proposal.id,
                problem_id=match.problem_id,
                citation_context=citation.surrounding_text
            ))
        else:
            # Strip unverifiable citation from rationale
            proposal.rationale = remove_citation(proposal.rationale, citation)
            log.warning(f"Unverifiable citation removed: {citation.text[:50]}...")
    
    await save_citations(verified)
```

### 2. Prompt Versioning

- Every generated artifact stores `prompt_version` field
- Prompt files are version-controlled in `prompts/` directory
- A/B testing: run same input through two prompt versions, compare output quality

### 3. Cost Tracking

- Every job records `token_usage` (prompt + completion tokens, USD cost)
- Dashboard shows: cost per evidence document, cost per proposal, daily/monthly totals
- Alerts if cost exceeds configurable threshold per job

### 4. Eval Harness

```python
# tests/eval/test_extraction_quality.py

GOLDEN_SET = [
    {
        "input": "I tried for 2 hours to configure permissions and just gave up...",
        "expected_problems": [
            {"statement_contains": "permissions", "severity": "high", "tags_contain": "permissions"}
        ]
    },
    # ... 50+ regression test cases
]

async def test_extraction_regression():
    for case in GOLDEN_SET:
        result = await extract_problems(case["input"])
        for expected in case["expected_problems"]:
            matching = [p for p in result.problems if expected["statement_contains"] in p.problem_statement.lower()]
            assert len(matching) >= 1, f"Missing expected problem: {expected}"
            assert matching[0].severity == expected["severity"]
```

### 5. Idempotency

- Re-extracting problems for the same evidence replaces previous extractions (not duplicates)
- Re-clustering overwrites previous clustering assignments
- Re-generating proposals creates a new `proposal_version`, preserving history

---

## Development Phases

### Phase 1: Evidence + Extraction (Weeks 1-4)

**Goal:** Upload transcripts, extract structured problems, prove LLM extraction quality.

| Week | Deliverable |
|------|------------|
| 1 | FastAPI project setup, Postgres + pgvector schema, Docker Compose |
| 2 | Evidence Service: upload, chunk, store with provenance |
| 3 | Extraction Service: LLM-based problem extraction with validation |
| 4 | Embedding Service: embed problems, similarity search endpoint |

**Exit Criteria:**
- Upload a transcript → get back structured problem mentions with quotes
- Quotes are verifiable against source text
- Similar problems are retrievable via embedding search

### Phase 2: Clustering + Proposals (Weeks 5-8)

**Goal:** See problems grouped into clusters, generate actionable feature proposals.

| Week | Deliverable |
|------|------------|
| 5 | Threshold clustering implementation + cluster summarization |
| 6 | Proposal Generator: cluster → feature proposal with citations |
| 7 | Citation verification pipeline + proposal versioning |
| 8 | Prioritization engine + roadmap endpoint |

**Exit Criteria:**
- Problems auto-cluster into labeled groups with top quotes
- Feature proposals are generated with verifiable citations
- Roadmap is ranked with explainable scores

### Phase 3: Task Trees + Polish (Weeks 9-12)

**Goal:** Complete pipeline from evidence to dev-ready tasks.

| Week | Deliverable |
|------|------------|
| 9 | Task Tree Generator: proposal → hierarchical tasks with acceptance criteria |
| 10 | Job queue (Celery/Redis) for all async LLM operations |
| 11 | Cost tracking, prompt versioning, eval harness |
| 12 | API polish, error handling, documentation, load testing |

**Exit Criteria:**
- End-to-end pipeline works: upload → problems → clusters → proposals → tasks → roadmap
- All LLM calls have cost tracking and prompt versioning
- Eval harness passes on golden set of test transcripts

### Phase 4: Production Hardening (Weeks 13-16)

| Week | Deliverable |
|------|------------|
| 13 | Authentication (API key → JWT migration path) |
| 14 | Rate limiting, request validation hardening |
| 15 | Database optimization (indexes, connection pooling, query plans) |
| 16 | Docker production config, health checks, monitoring |

---

## Deployment & Infrastructure

### Local Development

```yaml
# docker-compose.yml
services:
  api:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://nexus:nexus@db:5432/nexus
      - REDIS_URL=redis://redis:6379/0
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on: [db, redis]

  db:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    environment:
      - POSTGRES_DB=nexus
      - POSTGRES_USER=nexus
      - POSTGRES_PASSWORD=nexus
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  worker:
    build: ./backend
    command: celery -A app.worker worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://nexus:nexus@db:5432/nexus
      - REDIS_URL=redis://redis:6379/0
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on: [db, redis]

volumes:
  pgdata:
```

### Backend Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app + router registration
│   ├── config.py                   # Settings (Pydantic BaseSettings)
│   ├── database.py                 # Async SQLAlchemy + pgvector setup
│   ├── models/                     # SQLAlchemy ORM models
│   │   ├── evidence.py
│   │   ├── problems.py
│   │   ├── clusters.py
│   │   ├── proposals.py
│   │   ├── tasks.py
│   │   └── jobs.py
│   ├── schemas/                    # Pydantic request/response schemas
│   │   ├── evidence.py
│   │   ├── problems.py
│   │   ├── clusters.py
│   │   ├── proposals.py
│   │   ├── tasks.py
│   │   └── jobs.py
│   ├── services/                   # Business logic
│   │   ├── evidence_service.py
│   │   ├── extraction_service.py
│   │   ├── embedding_service.py
│   │   ├── clustering_service.py
│   │   ├── proposal_service.py
│   │   ├── task_tree_service.py
│   │   └── prioritization_service.py
│   ├── routers/                    # API route handlers
│   │   ├── evidence.py
│   │   ├── problems.py
│   │   ├── clusters.py
│   │   ├── proposals.py
│   │   ├── tasks.py
│   │   ├── jobs.py
│   │   └── roadmap.py
│   ├── llm/                        # LLM orchestration
│   │   ├── client.py               # Gemini/OpenAI client wrapper
│   │   ├── prompts.py              # Prompt registry + loading
│   │   └── cost_tracker.py         # Token usage + cost calculation
│   ├── worker.py                   # Celery worker + task definitions
│   └── utils/
│       ├── chunking.py             # Text chunking with offset tracking
│       ├── citations.py            # Citation extraction + verification
│       └── scoring.py              # Priority score calculation
├── prompts/                        # Versioned prompt templates
│   ├── extract_problems_v1.0.txt
│   ├── summarize_cluster_v1.0.txt
│   ├── generate_proposal_v1.0.txt
│   └── generate_tasks_v1.0.txt
├── migrations/                     # Alembic database migrations
│   ├── alembic.ini
│   └── versions/
├── tests/
│   ├── test_evidence.py
│   ├── test_extraction.py
│   ├── test_clustering.py
│   ├── test_proposals.py
│   ├── test_tasks.py
│   └── eval/
│       ├── golden_set.json         # Regression test data
│       └── test_extraction_quality.py
├── Dockerfile
├── requirements.txt
└── .env.example
```

### Key Dependencies

```
# requirements.txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
sqlalchemy[asyncio]>=2.0.25
asyncpg>=0.29.0
pgvector>=0.2.4
alembic>=1.13.0
celery[redis]>=5.3.0
redis>=5.0.0
google-generativeai>=0.4.0
httpx>=0.26.0
numpy>=1.26.0
umap-learn>=0.5.5
hdbscan>=0.8.33
python-multipart>=0.0.6
python-dotenv>=1.0.0
structlog>=24.1.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

---

## Migration from Existing Codebase

The existing `backend/` directory contains a ChromaDB-based RAG prototype. The new PM pipeline will:

1. **Coexist initially** — New services live in `app/` subdirectory alongside existing files
2. **Share config** — Reuse `config.py` patterns (API keys, environment variables)
3. **Preserve ChromaDB** — Document RAG features (RAPTOR, agent queries) continue using ChromaDB
4. **New data layer** — PM pipeline uses Postgres (structured data + vectors) separately
5. **Unified API** — Both systems served from same FastAPI instance, different route prefixes (`/api/v1/rag/` vs `/api/v1/pm/`)

---

## Summary

The backend is an **evidence-processing pipeline** with seven distinct services, each handling one step of the transformation:

```
Raw Evidence → Chunks → Problem Mentions → Embeddings → Clusters → Proposals → Tasks → Roadmap
```

Every step is:
- **Async** (LLM calls don't block the API)
- **Traceable** (provenance from roadmap item back to original quote)
- **Versioned** (prompt versions + proposal versions)
- **Costed** (token usage tracked per job)
- **Testable** (eval harness with golden set regression tests)

Build Phase 1 first. Validate extraction quality. Then proceed.

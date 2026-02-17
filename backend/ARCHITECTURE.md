# Nexus Backend — Architecture & Implementation Reference

> Complete technical documentation of everything built so far in the PM pipeline backend.

---

## Table of Contents

1. [Stack & Infrastructure](#stack--infrastructure)
2. [Entry Point & Startup](#entry-point--startup)
3. [Configuration](#configuration)
4. [Database Layer](#database-layer)
5. [Data Model (All Tables)](#data-model-all-tables)
6. [Stage 1: Evidence Ingestion](#stage-1-evidence-ingestion)
7. [Stage 2: Problem Extraction](#stage-2-problem-extraction)
8. [Stage 3: Embedding](#stage-3-embedding)
9. [Stage 4: Clustering](#stage-4-clustering)
10. [Stage 5: Proposals & Roadmap](#stage-5-proposals--roadmap)
11. [Stage 6: Similarity Search](#stage-6-similarity-search)
12. [Stage 7: Stats & Cost Tracking](#stage-7-stats--cost-tracking)
13. [Job System](#job-system)
14. [LLM Client](#llm-client)
15. [Middleware](#middleware)
16. [API Endpoint Reference](#api-endpoint-reference)
17. [Schemas (Pydantic)](#schemas-pydantic)
18. [Eval Harness](#eval-harness)
19. [Project Structure](#project-structure)
20. [Data Flow Summary](#data-flow-summary)

---

## Stack & Infrastructure

| Component | Technology | Details |
|-----------|-----------|---------|
| API Framework | FastAPI | Async-native, Pydantic validation, auto OpenAPI docs |
| Database | PostgreSQL + pgvector | Structured data + 768-dim vector embeddings in one DB |
| ORM | SQLAlchemy 2.0 (async) | AsyncSession via asyncpg driver |
| Migrations | Alembic | 2 migration files (Phase 1 + Phase 2 tables) |
| LLM Provider | Gemini 2.0 Flash | JSON generation + text-embedding-004 for vectors |
| Job Execution | FastAPI BackgroundTasks | In-memory, no Celery/Redis queue yet |
| Containerization | Docker Compose | API + Postgres (pgvector:pg16) + Redis containers |
| Rate Limiting | ~~In-memory token bucket~~ **In-memory sliding window log** | Per-IP, configurable via env vars |
| Auth | API key middleware | Built but not wired to any route (dev mode) |

### Docker Compose Services

```
api      →  FastAPI on port 8000
db       →  pgvector/pgvector:pg16 on port 5432
redis    →  redis:7-alpine on port 6379 (provisioned but not used by app yet)
```

---

## Entry Point & Startup

**File:** `app/main.py`

The FastAPI app is created with two middleware layers (rate limiting, CORS) and four routers. On startup:

1. `CREATE EXTENSION IF NOT EXISTS vector` — ensures pgvector is available
2. `Base.metadata.create_all` — auto-creates all SQLAlchemy-defined tables
3. All 8 ORM models are imported to register with the metadata

```python
app.include_router(evidence.router,          prefix="/api/v1", tags=["evidence"])
app.include_router(jobs.router,              prefix="/api/v1", tags=["jobs"])
app.include_router(problems.router,          prefix="/api/v1", tags=["problems"])
app.include_router(clusters_router.router,   prefix="/api/v1", tags=["clusters"])
```

CORS allows `localhost:3000` and `127.0.0.1:3000` (the Next.js frontend).

---

## Configuration

**File:** `app/config.py`

Uses Pydantic `BaseSettings` to load from `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | (required) | Postgres connection string; auto-normalized to `postgresql+asyncpg://` |
| `REDIS_URL` | `redis://localhost:6379/0` | Reserved for future Celery worker |
| `GEMINI_API_KEY` | (required) | Google AI API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Model for JSON generation |
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Model for embeddings (768 dims) |
| `CHUNK_MAX_TOKENS` | `500` | Max tokens per evidence chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between consecutive chunks |
| `API_KEYS` | `""` | Comma-separated API keys; empty = auth disabled |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window per IP |
| `RATE_LIMIT_WINDOW` | `60` | Window size in seconds |

The `normalize_db_url` validator converts `postgresql://` or `postgres://` prefixes to `postgresql+asyncpg://` so asyncpg works correctly (important for Supabase URLs).

---

## Database Layer

**File:** `app/database.py`

- **Engine:** `create_async_engine` with connection pooling (`pool_size=10`, `max_overflow=20`, `pool_recycle=300`**, `pool_pre_ping=True`**)
- **SSL:** Auto-detected ~~for Supabase URLs~~ **when `"supabase"` or `"ssl=require"` appears in the database URL** — creates an `ssl.SSLContext` with `check_hostname=False`, `verify_mode=CERT_NONE`. **Also strips `?ssl=require` / `&ssl=require` from the URL since asyncpg does not understand it as a query parameter.**
- **Session factory:** `async_sessionmaker` with `expire_on_commit=False`
- **Dependency:** `get_session()` is an async generator yielding `AsyncSession` for FastAPI `Depends()`

```python
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

---

## Data Model (All Tables)

### Entity Relationship

```
evidence (1) ──────< evidence_chunks (N)
                          │
                          │ (LLM extracts from chunks)
                          ▼
                    problem_mentions (N)
                          │
                          ├──── problem_embeddings (1:1)
                          │
                          ├────> cluster_memberships (N) >──── problem_clusters (1)
                          │                                         │
                          │                                         ▼
                          │                                  feature_proposals (N)
                          │                                         │
                          └──────────────────< proposal_citations (N)┘
```

### Table: `evidence`

**File:** `app/models/evidence.py`

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK, `gen_random_uuid()` |
| `title` | TEXT | NOT NULL |
| `source_type` | TEXT | NOT NULL — `interview`, `support_ticket`, `sales_note`, `survey`, `other` |
| `persona` | TEXT | Nullable |
| `segment` | TEXT | Nullable |
| `source_date` | DATE | Nullable |
| `raw_text` | TEXT | NOT NULL — full transcript/document |
| `metadata` | JSONB | Default `{}` — arbitrary key-value bag |
| `created_at` | TIMESTAMPTZ | `NOW()` |
| `updated_at` | TIMESTAMPTZ | `NOW()`, auto-updates on change |

**Relationships:**
- `chunks` → `EvidenceChunk` (cascade delete)
- `problem_mentions` → `ProblemMention` (cascade delete)

### Table: `evidence_chunks`

**File:** `app/models/evidence.py`

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `evidence_id` | UUID | FK → `evidence.id` (CASCADE) |
| `chunk_index` | INTEGER | 0-based position in document |
| `chunk_text` | TEXT | The chunk content |
| `start_offset` | INTEGER | Character offset in `raw_text` |
| `end_offset` | INTEGER | Character offset in `raw_text` |
| `token_count` | INTEGER | ~~Estimated token count (len/4)~~ **Nullable** — estimated token count (len/4) |
| `created_at` | TIMESTAMPTZ | `NOW()` |

**Relationships:**
- `evidence` → parent `Evidence`
- `problem_mentions` → `ProblemMention` (cascade delete)

### Table: `problem_mentions`

**File:** `app/models/problems.py`

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `evidence_id` | UUID | FK → `evidence.id` (CASCADE) |
| `chunk_id` | UUID | FK → `evidence_chunks.id` (CASCADE) |
| `problem_statement` | TEXT | Normalized problem description |
| `persona` | TEXT | Nullable — inherited from evidence if LLM doesn't extract one |
| `segment` | TEXT | Nullable — inherited from evidence if LLM doesn't extract one |
| `severity` | TEXT | `critical`, `high`, `medium`, `low` |
| `quote_text` | TEXT | Direct quote from source text |
| `quote_start` | INTEGER | Character offset within chunk (nullable if fuzzy matched) |
| `quote_end` | INTEGER | Character offset within chunk |
| `tags` | TEXT[] | Array of tags, e.g. `['pricing', 'onboarding']` |
| `extraction_job_id` | UUID | Nullable — reserved for future use |
| `prompt_version` | TEXT | Currently `"extract_problems_v1"` |
| `created_at` | TIMESTAMPTZ | `NOW()` |

**Relationships:**
- `evidence` → parent `Evidence`
- `chunk` → parent `EvidenceChunk`
- `embedding` → `ProblemEmbedding` (1:1, uselist=False)

### Table: `problem_embeddings`

**File:** `app/models/embeddings.py`

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `problem_id` | UUID | FK → `problem_mentions.id` (CASCADE), **UNIQUE** |
| `embedding` | Vector(768) | pgvector column — Gemini text-embedding-004 output |
| `model_version` | TEXT | Currently `"text-embedding-004"` |
| `created_at` | TIMESTAMPTZ | `NOW()` |

**Index:** HNSW on embedding column for fast approximate nearest neighbor search.

**Relationship:** `problem` → parent `ProblemMention`

### Table: `problem_clusters`

**File:** `app/models/clusters.py`

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `label` | TEXT | First 120 chars of first member's problem_statement |
| `summary` | TEXT | Nullable — reserved for LLM summarization (not implemented) |
| `centroid` | Vector(768) | Running mean of member embeddings |
| `threshold` | FLOAT | Similarity threshold used (default 0.75) |
| `mention_count` | INTEGER | Number of problems in this cluster |
| `tags` | TEXT[] | Empty by default |
| `metadata` | JSONB | Default `{}` |
| `created_at` | TIMESTAMPTZ | `NOW()` |
| `updated_at` | TIMESTAMPTZ | `NOW()`, auto-updates |

**Relationships:**
- `members` → `ClusterMembership` (cascade delete)
- `proposals` → `FeatureProposal` (cascade delete)

### Table: `cluster_memberships`

**File:** `app/models/clusters.py`

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `cluster_id` | UUID | FK → `problem_clusters.id` (CASCADE) |
| `problem_id` | UUID | FK → `problem_mentions.id` (CASCADE) |
| `similarity` | FLOAT | Cosine similarity to cluster centroid at time of assignment |
| `created_at` | TIMESTAMPTZ | `NOW()` |

**Relationships:**
- `cluster` → parent `ProblemCluster`
- `problem` → `ProblemMention`

### Table: `feature_proposals`

**File:** `app/models/clusters.py`

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `cluster_id` | UUID | FK → `problem_clusters.id` (CASCADE) |
| `title` | TEXT | NOT NULL |
| `description` | TEXT | NOT NULL |
| `priority_score` | FLOAT | Nullable — manually set, no formula |
| `impact` | TEXT | Nullable — free text (high/medium/low) |
| `effort` | TEXT | Nullable — free text (high/medium/low) |
| `version` | INTEGER | Default 1 — no versioning system uses this yet |
| `metadata` | JSONB | Default `{}` |
| `created_at` | TIMESTAMPTZ | `NOW()` |
| `updated_at` | TIMESTAMPTZ | `NOW()`, auto-updates |

**Relationships:**
- `cluster` → parent `ProblemCluster`
- `citations` → `ProposalCitation` (cascade delete)

### Table: `proposal_citations`

**File:** `app/models/clusters.py`

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `proposal_id` | UUID | FK → `feature_proposals.id` (CASCADE) |
| `problem_id` | UUID | FK → `problem_mentions.id` (CASCADE) |
| `relevance_note` | TEXT | Nullable — why this problem supports the proposal |
| `created_at` | TIMESTAMPTZ | `NOW()` |

**Relationships:**
- `proposal` → parent `FeatureProposal`
- `problem` → `ProblemMention`

---

## Stage 1: Evidence Ingestion

**Files:** `app/services/evidence_service.py`, `app/utils/chunking.py`, `app/routers/evidence.py`

### Trigger

`POST /api/v1/evidence` — JSON body with title, raw text, and optional metadata.

### Process

**1. Store raw evidence.** An `Evidence` row is inserted with a UUID primary key (generated by Postgres `gen_random_uuid()`). The full raw text is stored alongside all metadata.

**2. Chunk the text.** `chunk_text()` in `utils/chunking.py`:

- **Sentence splitting:** Regex `(?<=[.!?])\s+` splits on sentence-ending punctuation followed by whitespace. Each sentence gets character offsets (`start`, `end`) tracked against the original text.

- **Token estimation:** `len(text) // 4` — rough approximation of 1 token ≈ 4 characters for English.

- **Greedy accumulation:** Sentences are appended to a chunk until the next sentence would exceed `max_tokens` (500). A single oversized sentence becomes its own chunk — a chunk is never empty.

- **Overlap handling:** After building a chunk, the algorithm walks backward from the end counting sentences that fit within `overlap_tokens` (50). The next chunk starts that many sentences back. A safety `max(j - overlap_count, i + 1)` ensures at least 1 sentence of forward progress, preventing infinite loops.

- **Output per chunk:** `{index, text, start_offset, end_offset, token_count}`

**3. Store chunks.** Each chunk → `EvidenceChunk` row linked via `evidence_id`. Character offsets from the original `raw_text` are preserved for provenance tracing.

**4. Return.** Response includes the evidence UUID and chunk count. No extraction is triggered automatically.

### Cascade Behavior

Deleting an evidence record cascades to all its chunks, which cascades to all problem mentions extracted from those chunks.

### Query Endpoints

| Endpoint | Logic |
|----------|-------|
| `GET /evidence` | Paginated list with chunk counts (subquery count), filterable by `source_type`, `persona`, `segment`. Sorted by `created_at DESC`. |
| `GET /evidence/{id}` | Eager-loads all chunks via `selectinload(Evidence.chunks)`. Returns raw text + chunks. |
| `DELETE /evidence/{id}` | Deletes evidence + cascades to chunks + problem mentions. |

---

## Stage 2: Problem Extraction

**Files:** `app/services/extraction_service.py`, `app/routers/jobs.py`

### Trigger

`POST /api/v1/jobs/extract_problems` — body: `{"evidence_id": "...", "max_chunks": null}`. Returns HTTP 202 with a `job_id` immediately. Work runs via `BackgroundTasks`.

### Process

**1. Idempotency.** `_clear_existing_mentions()` deletes ALL existing `problem_mentions` for this evidence ID, then commits. Re-extraction replaces, never duplicates.

**2. Load chunks.** All `EvidenceChunk` rows for the evidence, ordered by `chunk_index ASC`. Optional `max_chunks` limit for testing.

**3. Concurrent LLM calls.** `asyncio.Semaphore(4)` limits to 4 concurrent chunks. `asyncio.gather()` processes all chunks in parallel (up to the semaphore limit).

For each chunk:

**3a. Build prompt:**
```
You are extracting customer problems from a transcript chunk.
Return valid JSON only, with this schema:
{
  "problems": [
    {
      "problem_statement": "string",
      "severity": "critical|high|medium|low",
      "quote_text": "direct quote from the chunk",
      "persona": "optional",
      "segment": "optional",
      "tags": ["tag1", "tag2"]
    }
  ]
}

If no problems are present, return {"problems": []}.

Chunk:
{chunk_text}
```

**3b. Call with retry.** `_call_with_retry()` wraps `client.generate_json()`:
- Up to 3 attempts (`MAX_RETRIES = 3`)
- Exponential backoff: ~~1s → 2s → 4s~~ **1s → 2s** (3 attempts = 2 retry sleeps; the 4s value is never reached)
- The call runs via `asyncio.to_thread()` because the Gemini SDK is synchronous
- On final failure, the exception propagates and the chunk yields 0 problems (caught by the outer try/except)

**3c. Parse & validate.** The raw LLM response goes through:
1. `_parse_json_response()` in the LLM client — strips markdown fences (```` ```json ... ``` ````), finds the JSON object boundaries (`{` to `}`), calls `json.loads`
2. Pydantic validation: `LLMProblemsResponse.model_validate(raw)` enforces the schema — `problems` must be a list of `ProblemMentionCreate` objects with valid `severity` literals

**4. Quote verification — the critical quality gate.** For each extracted problem, `_build_problem_mention()` calls `_find_quote_offsets(chunk_text, quote_text)`:

- **Exact match:** `chunk_text.find(quote_text)`. If found → exact character offsets, verified = True.

- **Fuzzy sliding window:** If no exact match, a window roughly the size of the quote slides across the chunk with step size `max(1, quote_len // 10)`. At each position, `fuzz.partial_ratio(quote.lower(), window.lower())` scores the match (0–100). The highest-scoring position is tracked.

- **Threshold decision:** If best fuzzy score ≥ 70 (`FUZZY_MATCH_THRESHOLD`), the quote is accepted with the best-fit offsets. If below 70, the **entire problem mention is dropped** — it returns `None` and a warning is logged. This prevents hallucinated or paraphrased quotes from entering the database.

**5. Metadata inheritance.** If the LLM didn't extract a persona or segment, the function uses the parent evidence's persona/segment as a fallback. The `prompt_version` string `"extract_problems_v1"` is stamped on every mention.

**6. Persist.** All verified `ProblemMention` objects are `session.add_all()`'d and committed in a single transaction.

### Constants

```python
PROMPT_VERSION = "extract_problems_v1"
MAX_CONCURRENCY = 4
MAX_RETRIES = 3
FUZZY_MATCH_THRESHOLD = 70
```

---

## Stage 3: Embedding

**File:** `app/services/embeddings_service.py`

### Trigger

`POST /api/v1/jobs/embed_problems` — optional body: `{"limit": null}`. Same background-task pattern.

### Process

**1. Find unembedded problems.** A LEFT JOIN query finds all `ProblemMention` rows that have no corresponding `ProblemEmbedding` row:
```sql
SELECT pm.* FROM problem_mentions pm
LEFT JOIN problem_embeddings pe ON pm.id = pe.problem_id
WHERE pe.id IS NULL
ORDER BY pm.created_at DESC
```
This is naturally incremental — only new/unembedded problems are processed.

**2. Build embedding text.** Each problem is embedded as:
```
{problem_statement}

Quote: {quote_text}
```
Combining both the normalized statement and the raw quote gives richer semantic signal than either alone.

**3. Concurrent calls.** Same `Semaphore(4)` + `_call_with_retry(3)` pattern. Each call goes to `GeminiClient.embed_text()`:
```python
genai.embed_content(
    model=settings.gemini_embedding_model,  # default: "text-embedding-004"
    content=text,
    task_type="RETRIEVAL_DOCUMENT",
)
```
Returns a 768-dimension float vector.

**4. Store.** Each vector becomes a `ProblemEmbedding` row. The `problem_id` column has a `UNIQUE` constraint (1:1 with problem mentions). The `model_version` is recorded as `"text-embedding-004"`.

**5. Return.** The list of successfully embedded `problem_id` UUIDs is returned. Failed embeddings are logged and skipped (don't block the batch).

---

## Stage 4: Clustering

**File:** `app/services/cluster_service.py`

### Trigger

`POST /api/v1/clusters/run?threshold=0.75` — runs **synchronously** (not a background task). Blocks until all embeddings are clustered.

### Algorithm: Greedy Threshold Clustering

**1. Load all embeddings.** Every `ProblemEmbedding` row is fetched with its `ProblemMention` eagerly loaded via `selectinload`.

**2. Iterate and assign.** For each embedding vector:

```python
for pe in rows:
    vec = np.array(pe.embedding, dtype=np.float32)

    # Compare to every existing cluster centroid
    for cluster in clusters:
        centroid = np.array(cluster.centroid, dtype=np.float32)
        sim = dot(vec, centroid) / (||vec|| * ||centroid|| + 1e-9)
        # Track the best match

    if best_sim >= threshold:
        # Add to existing cluster
    else:
        # Create new cluster
```

**3. Centroid update (running mean).** When a problem joins a cluster:
```python
old_centroid = np.array(cluster.centroid)
n = cluster.mention_count  # already incremented
new_centroid = old_centroid + (vec - old_centroid) / n
cluster.centroid = new_centroid.tolist()
```
This avoids recomputing the centroid from scratch each time.

**4. New cluster creation.** If no cluster exceeds the threshold:
- Label = first 120 characters of the problem's `problem_statement`
- Centroid = the problem's embedding vector
- `mention_count` starts at 1
- A `ClusterMembership` row is created with `similarity=1.0`

**5. Persist.** All clusters and memberships are committed in a single transaction.

### Limitations

- No LLM-generated labels or summaries — labels are just truncated problem statements
- No HDBSCAN or UMAP — simple greedy assignment
- Doesn't clear old clusters before re-running (would duplicate)
- Order-dependent — results vary based on the iteration order of embeddings
- **⚠ Design Issue:** Endpoint returns HTTP 202 (Accepted) despite executing synchronously — 202 conventionally implies async/background processing

### Query Endpoints

| Endpoint | Logic |
|----------|-------|
| `GET /clusters` | Paginated, sorted by `mention_count DESC` (biggest clusters first) |
| `GET /clusters/{id}` | Eager-loads `members` (ClusterMembership list) and `proposals` (FeatureProposal list) |

---

## Stage 5: Proposals & Roadmap

**File:** `app/services/cluster_service.py` (proposals are in the same file as clustering)

### Proposals (Manual Creation Only)

`POST /api/v1/proposals` — creates a proposal by hand. No LLM generation.

**Input:**
```json
{
  "cluster_id": "uuid",
  "title": "Guided Onboarding Wizard",
  "description": "Step-by-step setup flow...",
  "priority_score": 42.5,
  "impact": "high",
  "effort": "medium"
}
```

The `FeatureProposal` model has a `version` field (defaults to 1) and a `metadata` JSONB column, but no versioning system actually increments or snapshots them.

### Citations

`POST /api/v1/proposals/{id}/citations?problem_id=...&relevance_note=...`

Links a proposal to a specific problem mention via a `ProposalCitation` row. The `relevance_note` is optional free text explaining why this problem supports the proposal.

### Roadmap

`GET /api/v1/roadmap`

- Joins `feature_proposals` with `problem_clusters` on `cluster_id`
- Orders by `priority_score DESC NULLS LAST`
- Returns: proposal object, cluster label, mention count, priority score
- No scoring formula, no breakdown, no weight adjustment — just the manually-set float

---

## Stage 6: Similarity Search

**File:** `app/routers/problems.py`, `app/services/problems_service.py`

### Endpoint

`GET /api/v1/problems/similar?text=...&limit=10&min_score=0.0`

### Process

1. The query text is embedded in real-time via `GeminiClient.embed_text()` (same model used for stored embeddings)
2. pgvector's `<=>` cosine distance operator performs kNN search:
```sql
SELECT pm.*, (1 - (pe.embedding <=> query_vec)) AS similarity
FROM problem_embeddings pe
JOIN problem_mentions pm ON pm.id = pe.problem_id
ORDER BY pe.embedding <=> query_vec ASC
LIMIT :limit
```
3. Results below `min_score` are post-filtered
4. Returns problem mentions ranked by similarity (1.0 = identical, 0.0 = orthogonal)

---

## Stage 7: Stats & Cost Tracking

### Problem Stats

`GET /api/v1/problems/stats?persona=...&severity=...&tag=...`

Runs four SQL aggregation queries:
1. **Total count** — `COUNT(*)` with optional filters
2. **By severity** — `GROUP BY severity`
3. **By persona** — `GROUP BY persona WHERE persona IS NOT NULL`
4. **By tag** — `SELECT unnest(tags) AS tag, COUNT(*) GROUP BY tag` — uses PostgreSQL `unnest()` to explode the tags array

All queries respect the same optional filters (persona, severity, tag).

### LLM Cost Tracking

**In-memory only** — lost on server restart.

| Endpoint | Returns |
|----------|---------|
| `GET /api/v1/llm/costs` | `{total_calls, total_cost_usd, total_input_tokens, total_output_tokens, by_model: {model: cost}}` |
| `GET /api/v1/llm/calls` | Array of every `LLMCallRecord` — model, operation, prompt_version, input/output tokens, latency_ms, cost_usd, timestamp, error |

---

## Job System

**File:** `app/routers/jobs.py`

### Architecture

Jobs run via FastAPI `BackgroundTasks` — no Celery, no Redis, no persistent queue. Job state is stored in an **in-memory dict** (`_job_store`) protected by an `asyncio.Lock`.

### Job Lifecycle

```
POST /jobs/extract_problems  →  job created (pending)
                                    │
                              BackgroundTasks.add_task()
                                    │
                              _run_extract_job()
                                    │
                              status = "running"
                                    │
                         ┌──── success ────┐── failure ──┐
                         ▼                               ▼
                   status = "completed"          status = "failed"
                   result_count = N              error = "..."
```

### Job Record Schema

```python
{
    "job_id": UUID,
    "status": "pending" | "running" | "completed" | "failed",
    "job_type": "extract_problems" | "embed_problems",
    "created_at": datetime,
    "started_at": datetime | None,
    "finished_at": datetime | None,
    "error": str | None,
    "result_count": int | None,
}
```

### Polling

`GET /api/v1/jobs/{job_id}/status` — returns the current job state. The frontend polls this every 2 seconds.

### Limitations

- **Not persistent** — all jobs lost on server restart
- **No progress tracking** — only pending/running/completed/failed (no percentage)
- **Only 2 job types** — `extract_problems` and `embed_problems`
- **No job for clustering** — `POST /clusters/run` executes synchronously

---

## LLM Client

**File:** `app/llm/client.py`

### GeminiClient

Singleton pattern via `get_client()` — initialized on first use.

**`generate_json(prompt, prompt_version=None) → dict`**

1. Calls `self.model.generate_content(prompt)` (synchronous Gemini SDK)
2. Extracts token counts from `response.usage_metadata` — falls back to `len(text)//4` estimate
3. Estimates cost via pricing table
4. Creates and appends an `LLMCallRecord` to in-memory `_call_log`
5. Parses JSON from the response via `_parse_json_response()`

**`embed_text(text) → list[float]`**

1. Calls `genai.embed_content(model="text-embedding-004", content=text, task_type="RETRIEVAL_DOCUMENT")`
2. Returns the 768-dim vector
3. Logs the call record with cost estimate

### JSON Response Parsing

`_parse_json_response(text)`:
1. Strip leading/trailing whitespace
2. If wrapped in triple backticks, strip them and the `json` language tag
3. Find the first `{` and last `}` in the string
4. `json.loads()` that substring
5. Raise `ValueError` if no JSON object found

### Pricing Table

```python
{
    "gemini-2.0-flash":    {"input": 0.000075, "output": 0.0003},    # per 1K tokens
    "text-embedding-004":  {"input": 0.000025, "output": 0.0},
}
```

### LLMCallRecord Fields

| Field | Type |
|-------|------|
| `model` | str |
| `operation` | `"generate_json"` or `"embed_text"` |
| `prompt_version` | str or None |
| `input_tokens` | int |
| `output_tokens` | int |
| `latency_ms` | float |
| `cost_usd` | float |
| `timestamp` | datetime (UTC) |
| `error` | str or None |

---

## Middleware

### Rate Limiting

**File:** `app/middleware/rate_limit.py`

- ~~In-memory token bucket~~ **In-memory sliding window log** per client IP — stores request timestamps and counts within window
- Default: 60 requests per 60-second window
- Returns HTTP 429 when exceeded
- Applied globally to all routes

### Authentication

**File:** `app/middleware/auth.py`

- API key auth via `X-API-Key` header
- If `API_KEYS` env var is empty, auth is disabled (dev mode)
- Provides a `RequireAuth` FastAPI dependency
- **Not currently applied to any route** — all endpoints are open

---

## API Endpoint Reference

### Evidence

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/api/v1/evidence` | Upload new evidence | 201 + `EvidenceResponse` |
| `GET` | `/api/v1/evidence` | List evidence (paginated) | `EvidenceListResponse` |
| `GET` | `/api/v1/evidence/{id}` | Evidence detail + chunks | `EvidenceDetailResponse` |
| `DELETE` | `/api/v1/evidence/{id}` | Delete evidence (cascades) | 204 |

### Problems

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET` | `/api/v1/problems` | List problems (filterable) | `ProblemMentionListResponse` |
| `GET` | `/api/v1/problems/{id}` | Problem detail | `ProblemMentionResponse` |
| `GET` | `/api/v1/problems/similar?text=...` | Similarity search | `SimilarProblemsResponse` |
| `GET` | `/api/v1/problems/stats` | Aggregate stats | `{total, by_severity, by_persona, by_tag}` |

### Jobs

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/api/v1/jobs/extract_problems` | Trigger extraction | 202 + `JobResponse` |
| `POST` | `/api/v1/jobs/embed_problems` | Trigger embedding | 202 + `JobResponse` |
| `GET` | `/api/v1/jobs/{id}/status` | Poll job status | `JobStatusResponse` |

### Clusters

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/api/v1/clusters/run?threshold=0.75` | Run clustering | 202 + `{clusters_created}` |
| `GET` | `/api/v1/clusters` | List clusters | `{items, total, page, ...}` |
| `GET` | `/api/v1/clusters/{id}` | Cluster detail | `ClusterDetailResponse` |

### Proposals

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/api/v1/proposals` | Create proposal (manual) | 201 + `ProposalResponse` |
| `POST` | `/api/v1/proposals/{id}/citations` | Add citation | `{id, proposal_id, problem_id}` |

### Roadmap

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET` | `/api/v1/roadmap` | Ranked proposals | `RoadmapResponse` |

### LLM / Observability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/llm/costs` | Cost summary (in-memory) |
| `GET` | `/api/v1/llm/calls` | Full call log (in-memory) |
| `GET` | `/api/v1/health` | `{"status": "ok"}` |

---

## Schemas (Pydantic)

### Evidence Schemas (`app/schemas/evidence.py`)

| Schema | Purpose |
|--------|---------|
| `EvidenceCreate` | POST body — title, source_type, persona, segment, source_date, raw_text, metadata |
| `EvidenceResponse` | Standard response — id, title, source_type, persona, segment, source_date, chunk_count, created_at |
| `EvidenceDetailResponse` | Extends `EvidenceResponse` with `raw_text` + `chunks[]` |
| `EvidenceChunkResponse` | id, chunk_index, chunk_text, start_offset, end_offset, token_count |
| `EvidenceListResponse` | `{items[], total, page, per_page, total_pages}` |

### Problem Schemas (`app/schemas/problems.py`)

| Schema | Purpose |
|--------|---------|
| `ExtractProblemsRequest` | Job trigger — evidence_id, max_chunks |
| `ProblemMentionCreate` | LLM output validation — problem_statement, severity, quote_text, persona, segment, tags |
| `LLMProblemsResponse` | Wrapper — `{problems: ProblemMentionCreate[]}` |
| `ProblemMentionResponse` | API response — all fields + id, evidence_id, chunk_id, created_at |
| `ProblemMentionListResponse` | Paginated — `{items[], total, page, per_page, total_pages}` |
| `SimilarProblemResult` | `{problem: ProblemMentionResponse, score: float}` |
| `SimilarProblemsResponse` | `{query_text, results: SimilarProblemResult[]}` |

### Cluster Schemas (`app/schemas/clusters.py`)

| Schema | Purpose |
|--------|---------|
| `ClusterResponse` | id, label, summary, threshold, tags, mention_count, timestamps |
| `ClusterDetailResponse` | Extends with `members[]` + `proposals[]` |
| `ClusterMemberResponse` | id, problem_id, similarity |
| `ProposalCreate` | cluster_id, title, description, priority_score, impact, effort |
| `ProposalResponse` | All fields + id, version, timestamps |
| `RoadmapItem` | `{proposal, cluster_label, mention_count, priority_score}` |
| `RoadmapResponse` | `{items: RoadmapItem[], total}` |

### Job Schemas (`app/schemas/jobs.py`)

| Schema | Purpose |
|--------|---------|
| `JobResponse` | `{job_id, status}` — returned on job creation |
| `JobStatusResponse` | All fields — job_id, status, job_type, created_at, started_at, finished_at, error, result_count |

### Embedding Schemas (`app/schemas/embeddings.py`)

| Schema | Purpose |
|--------|---------|
| `EmbedProblemsRequest` | `{limit: int | None}` |

---

## Eval Harness

**Files:** `app/eval/harness.py`, `app/eval/golden_set.json`

A CLI-runnable evaluation framework for testing extraction quality:

1. Loads test cases from `golden_set.json` (currently only 2 entries)
2. Runs extraction on each test case's input text
3. Fuzzy-matches extracted problems against expected problems
4. Reports precision, recall, and F1 scores

The golden set is minimal — the strategy doc targets 50+ regression test entries.

---

## Project Structure

```
backend/
├── alembic.ini                          # Alembic config
├── Dockerfile                           # Dev container config
├── Dockerfile.prod                      # Production container config
├── requirements.txt                     # Python dependencies
├── run.ps1                              # PowerShell launch script
├── alembic/
│   ├── env.py                           # Alembic environment
│   ├── script.py.mako                   # Migration template
│   └── versions/
│       ├── 001_phase1_tables.py         # Evidence + chunks + problems + embeddings
│       └── 002_phase2_clusters.py       # Clusters + memberships + proposals + citations
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI app, startup, router registration
│   ├── config.py                        # Pydantic BaseSettings
│   ├── database.py                      # Async SQLAlchemy engine + session
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── golden_set.json              # 2 test entries
│   │   └── harness.py                   # Extraction quality eval
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py                    # GeminiClient + LLMCallRecord + cost tracking
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py                      # API key auth (not wired)
│   │   └── rate_limit.py                # Token bucket rate limiter
│   ├── models/
│   │   ├── __init__.py                  # Re-exports all 8 models
│   │   ├── evidence.py                  # Evidence + EvidenceChunk
│   │   ├── problems.py                  # ProblemMention
│   │   ├── embeddings.py                # ProblemEmbedding (pgvector)
│   │   └── clusters.py                  # ProblemCluster + ClusterMembership + FeatureProposal + ProposalCitation
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── evidence.py                  # CRUD endpoints
│   │   ├── problems.py                  # List, detail, similar, stats
│   │   ├── jobs.py                      # Extract + embed jobs, polling, LLM cost endpoints
│   │   └── clusters.py                  # Clustering, proposals, citations, roadmap
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── evidence.py                  # Request/response models
│   │   ├── problems.py                  # Request/response + LLM output validation
│   │   ├── clusters.py                  # Cluster + proposal + roadmap schemas
│   │   ├── embeddings.py                # EmbedProblemsRequest
│   │   └── jobs.py                      # JobResponse + JobStatusResponse
│   ├── services/
│   │   ├── __init__.py
│   │   ├── evidence_service.py          # Ingest, chunk, store, list, detail, delete
│   │   ├── extraction_service.py        # LLM extraction + fuzzy quote verification
│   │   ├── embeddings_service.py        # Vector embedding generation
│   │   ├── cluster_service.py           # Threshold clustering + proposals + roadmap
│   │   └── problems_service.py          # List, filter, similarity search, stats
│   ├── utils/
│   │   ├── __init__.py
│   │   └── chunking.py                  # Sentence-boundary-aware text chunking
│   └── tests/
│       └── test_db.py                   # Basic DB connectivity test
```

---

## Data Flow Summary

```
POST /evidence
  └─→ evidence row + evidence_chunks rows
          │
POST /jobs/extract_problems
  └─→ (background) Gemini generates JSON per chunk
          │
          ├─→ Pydantic validates LLM output
          ├─→ Fuzzy quote verification (thefuzz, threshold 70)
          ├─→ Drops hallucinated quotes
          └─→ problem_mentions rows (with offsets, tags, severity)
                  │
POST /jobs/embed_problems
  └─→ (background) Gemini text-embedding-004
          └─→ problem_embeddings rows (768-dim vectors, HNSW indexed)
                  │
POST /clusters/run
  └─→ (sync) numpy cosine similarity, greedy threshold assignment
          ├─→ problem_clusters rows (with running-mean centroids)
          └─→ cluster_memberships rows (with similarity scores)
                  │
POST /proposals
  └─→ (manual) feature_proposals row
          │
POST /proposals/{id}/citations
  └─→ proposal_citations row (links proposal ↔ problem)
          │
GET /roadmap
  └─→ proposals JOIN clusters, sorted by priority_score DESC
```

### Provenance Chain

Every item in the roadmap traces all the way back to the original text:

```
Roadmap position
  → Feature Proposal (title, description, score)
    → Proposal Citation (relevance_note)
      → Problem Mention (statement, severity, quote, tags)
        → Evidence Chunk (text, character offsets)
          → Evidence (raw_text, title, source metadata)
```

This is the core architectural principle: **no claim without a traceable citation chain**.

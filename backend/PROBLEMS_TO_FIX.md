# Nexus Backend — Problems to Fix

> Comprehensive bug list, code issues, and architectural concerns.  
> Ranked by severity: 🔴 Critical → 🟠 Major → 🟡 Minor → 🔵 Info/Optimization  
> Links point to exact file + line for easy navigation.

---

## Table of Contents

1. [Documentation Bugs (from ARCHITECTURE.md review)](#1-documentation-bugs)
2. [Code Bugs & Defects](#2-code-bugs--defects)
3. [Design & Architecture Issues](#3-design--architecture-issues)
4. [Optimizations & Cleanup](#4-optimizations--cleanup)

---

## 1. Documentation Bugs

_Already fixed in ARCHITECTURE.md with tracked changes. Listed here for completeness._

| # | Severity | File | Issue |
|---|----------|------|-------|
| D1 | 🟠 | `ARCHITECTURE.md` | Rate limiter mislabeled as "token bucket" — actually a sliding window log |
| D2 | 🟠 | `ARCHITECTURE.md` | Retry backoff described as "1s → 2s → 4s" — actually "1s → 2s" (3 attempts = 2 sleeps) |
| D3 | 🟡 | `ARCHITECTURE.md` | `evidence_chunks.token_count` not marked as nullable |
| D4 | 🟡 | `ARCHITECTURE.md` | `pool_pre_ping=True` omitted from database layer description |
| D5 | 🟡 | `ARCHITECTURE.md` | SSL detection description too narrow (also triggers on `ssl=require`) |
| D6 | 🟡 | `ARCHITECTURE.md` | `embed_content` model param shown hardcoded instead of configurable |

---

## 2. Code Bugs & Defects

### 🔴 CRITICAL

#### C1 — Clustering never clears old data before re-run (duplicate clusters)
- **File:** [app/services/cluster_service.py](app/services/cluster_service.py#L30-L101)
- **Problem:** `run_threshold_clustering()` loads all embeddings and creates new clusters, but never deletes existing `problem_clusters` or `cluster_memberships` rows first. Every call to `POST /clusters/run` appends new duplicates on top of old ones.
- **Impact:** Database fills with duplicate clusters. The `mention_count` in the roadmap becomes meaningless. Every re-cluster doubles the mess.
- **Fix:** Add `DELETE FROM cluster_memberships` + `DELETE FROM problem_clusters` at the top of the function before creating new clusters (same pattern as `_clear_existing_mentions` in extraction).

#### C2 — Extraction re-run deletes mentions but orphans their embeddings
- **File:** [app/services/extraction_service.py](app/services/extraction_service.py#L162-L167)
- **Problem:** `_clear_existing_mentions()` deletes `problem_mentions` for re-extraction, but `problem_embeddings` has a CASCADE on `problem_mentions.id`. While CASCADE *does* delete the embeddings, this only works because of the FK ON DELETE CASCADE. However, if anyone later removes the CASCADE or changes the FK, embeddings will be orphaned. More importantly, the old **cluster_memberships** referencing those problem IDs will ALSO cascade-delete, silently destroying cluster data on re-extraction without the user realizing clusters need re-running.
- **Impact:** Re-extracting evidence silently invalidates clusters. No warning to the user.
- **Fix:** Either (a) warn/prevent re-extraction if clusters exist, or (b) delete affected clusters too, or (c) at minimum log a prominent warning.

#### C3 — `on_event("startup")` is deprecated in modern FastAPI
- **File:** [app/main.py](app/main.py#L40-L52)
- **Problem:** `@app.on_event("startup")` is deprecated since FastAPI 0.103+. Should use `lifespan` context manager instead.
- **Impact:** Will emit deprecation warnings immediately. Will break in a future FastAPI version.
- **Fix:** Convert to `@asynccontextmanager async def lifespan(app): ...` pattern.

#### C4 — `get_session` is an async generator but not typed as one
- **File:** [app/database.py](app/database.py#L36-L38)
- **Problem:** `get_session()` has return type `AsyncSession` but it's actually an `AsyncGenerator[AsyncSession, None]` because it uses `yield`. While FastAPI's `Depends()` handles this correctly at runtime, the type annotation is technically wrong and will confuse type checkers and IDE navigation.
- **Impact:** Static analysis tools (mypy, pyright) will flag incorrect types.
- **Fix:** Change return type to `AsyncGenerator[AsyncSession, None]` and import from `collections.abc`.

### 🟠 MAJOR

#### M1 — `asyncio.Lock` created at module level in jobs.py — broken across workers
- **File:** [app/routers/jobs.py](app/routers/jobs.py#L18)
- **Problem:** `_job_lock = asyncio.Lock()` is created at module import time, *before* any event loop exists. In Python 3.10+, `asyncio.Lock()` no longer binds to a running loop at creation time (fixed in 3.10), so this works with a single uvicorn worker. But with multiple workers (e.g. `uvicorn --workers 4`), each worker gets its own `_job_store` and `_job_lock` — jobs created in one worker are invisible to others.
- **Impact:** Job status polling returns 404 if the polling request is routed to a different worker than the one that created the job. Multi-worker deployments are broken.
- **Fix:** Move job storage to Redis or the database. The Redis container is already provisioned in docker-compose but never used.

#### M2 — No validation that `cluster_id` exists before creating a proposal
- **File:** [app/services/cluster_service.py](app/services/cluster_service.py#L148-L163)
- **Problem:** `create_proposal()` creates a `FeatureProposal` with the given `cluster_id` without checking if that cluster exists. If the cluster_id is invalid, the DB will raise an IntegrityError (FK violation) — which is returned as an unhandled 500 Internal Server Error.
- **Impact:** Users see a raw 500 error instead of a clear "cluster not found" 404.
- **Fix:** Check `session.get(ProblemCluster, cluster_id)` first, raise 404.

#### M3 — No validation that `proposal_id` or `problem_id` exists before creating a citation
- **File:** [app/services/cluster_service.py](app/services/cluster_service.py#L166-L177)
- **Problem:** Same issue as M2 — `add_citation()` blindly inserts without verifying FK targets exist.
- **Impact:** 500 on invalid IDs instead of 404.
- **Fix:** Verify both IDs exist before insert.

#### M4 — `_clear_existing_mentions` commits in the middle of a larger transaction
- **File:** [app/services/extraction_service.py](app/services/extraction_service.py#L162-L167)
- **Problem:** `_clear_existing_mentions()` does `await session.commit()` which commits the DELETE *before* new mentions are extracted. If the LLM calls then fail, the evidence has zero problem mentions — the old data is gone and no new data replaced it.
- **Impact:** Partial data loss on extraction failure. The user must re-run extraction manually.
- **Fix:** Use `flush()` instead of `commit()`, and let the outer function commit everything atomically.

#### M5 — `extract_problems_for_evidence` does both `flush()` and `commit()` redundantly
- **File:** [app/services/extraction_service.py](app/services/extraction_service.py#L142-L145)
- **Problem:** After `session.add_all(created_mentions)`, the code calls both `await session.flush()` and `await session.commit()`. The `flush()` is unnecessary before a `commit()` — commit already flushes.
- **Impact:** No functional bug, but indicates confusion about SQLAlchemy's flush/commit semantics. Removing `flush()` is cleaner.
- **Fix:** Remove the `flush()` call.

#### M6 — `test_db.py` queries wrong table names
- **File:** [app/tests/test_db.py](app/tests/test_db.py#L55-L56)
- **Problem:** The test script checks for tables named `"clusters"` and `"tasks"`, but the actual table names are `"problem_clusters"` and there is no `"tasks"` table. These will always silently skip.
- **Impact:** The connectivity test doesn't actually verify that the real tables exist.
- **Fix:** Change `"clusters"` to `"problem_clusters"`, remove `"tasks"`.

#### M7 — Similarity search uses `LIMIT` before post-filtering by `min_score`
- **File:** [app/services/problems_service.py](app/services/problems_service.py#L76-L84)
- **Problem:** The SQL query applies `LIMIT` first, then the Python code filters out results below `min_score`. If (say) limit=10 and 8 of the top 10 are below min_score, the user gets only 2 results even though there might be more qualifying results beyond position 10.
- **Impact:** Users may get fewer results than requested when using `min_score`.
- **Fix:** Either apply `min_score` as a SQL `WHERE` clause, or fetch more rows than the limit and trim after filtering.

#### M8 — `import asyncio` inside function body in problems router
- **File:** [app/routers/problems.py](app/routers/problems.py#L37)
- **Problem:** `import asyncio` is placed inside `similar_problems_endpoint()` instead of at the top of the file. While this works, it's unconventional and suggests the import was added hastily.
- **Impact:** Minor code smell. Slight performance hit on first call (negligible).
- **Fix:** Move `import asyncio` to the top of the file.

#### M9 — `source_type` in Evidence model is unconstrained `Text`, but schema uses a `Literal`
- **File:** [app/models/evidence.py](app/models/evidence.py#L19) vs [app/schemas/evidence.py](app/schemas/evidence.py#L7)
- **Problem:** The Pydantic schema correctly constrains `source_type` to `Literal["interview", "support_ticket", "sales_note", "survey", "other"]`, but the database column is just `Text` with no CHECK constraint. Data entering through other paths (direct SQL, migrations, other services) can insert invalid source types.
- **Impact:** Database can have invalid source_type values that the API layer would reject if queried.
- **Fix:** Add a CHECK constraint in the migration/model, or add a DB-level enum.

#### M10 — `severity` in ProblemMention model is unconstrained `Text`
- **File:** [app/models/problems.py](app/models/problems.py#L31)
- **Problem:** Same issue as M9 — the schema constrains severity to `"critical|high|medium|low"` but the database column is just `Text`. LLM output that sneaks through validation (e.g., a schema change) could write invalid severities.
- **Impact:** Low risk currently since Pydantic validates LLM output, but no defense-in-depth.
- **Fix:** Add CHECK constraint at DB level.

### 🟡 MINOR

#### m1 — `ExtractProblemsResponse` schema is defined but never used
- **File:** [app/schemas/problems.py](app/schemas/problems.py#L15-L18)
- **Problem:** `ExtractProblemsResponse` is defined with `evidence_id`, `extracted_count`, `problems` fields, but no router or service ever returns it.
- **Impact:** Dead code.
- **Fix:** Remove it, or wire it up if extraction should return problem details.

#### m2 — `EmbedProblemsResponse` schema is defined but never used
- **File:** [app/schemas/embeddings.py](app/schemas/embeddings.py#L9-L11)
- **Problem:** `EmbedProblemsResponse` with `embedded_count` and `problem_ids` is defined but no endpoint returns it.
- **Impact:** Dead code.
- **Fix:** Remove or use it.

#### m3 — `ClusterCreate` schema is defined but never used
- **File:** [app/schemas/clusters.py](app/schemas/clusters.py#L19-L20)
- **Problem:** Inherits from `ClusterBase` but is never referenced anywhere — clusters are only created via the service function.
- **Impact:** Dead code.
- **Fix:** Remove.

#### m4 — `ProposalDetailResponse` and `CitationResponse` schemas are defined but never used
- **File:** [app/schemas/clusters.py](app/schemas/clusters.py#L65-L72)
- **Problem:** These schemas exist but no endpoint returns them. Proposal detail with citations is not exposed.
- **Impact:** Dead code.
- **Fix:** Either expose a `GET /proposals/{id}` detail endpoint that uses them, or remove.

#### m5 — `Iterable` type hint used for `.scalars().all()` return values
- **File:** [app/services/extraction_service.py](app/services/extraction_service.py#L104), [app/services/embeddings_service.py](app/services/embeddings_service.py#L36)
- **Problem:** Both files annotate the result of `.scalars().all()` as `Iterable[...]`, but `.all()` returns a `Sequence` (specifically a `list`). Using `Iterable` is less specific and prevents using `len()` without type errors.
- **Impact:** Misleading type annotation.
- **Fix:** Use `Sequence[...]` or just `list[...]`.

#### m6 — `Dockerfile.prod` exists but contents unknown / likely outdated
- **File:** `Dockerfile.prod`
- **Problem:** A production Dockerfile exists but there's no indication it's being tested or kept in sync with `Dockerfile`.
- **Impact:** May not work when actually needed for production.
- **Fix:** Verify it, or remove if not maintained.

#### m7 — Overlap handling in chunking produces slightly wrong offsets
- **File:** [app/utils/chunking.py](app/utils/chunking.py#L92-L97)
- **Problem:** When chunks overlap, the overlapping sentences are re-joined with `" ".join()` which may produce different whitespace than the original text. The `start_offset` and `end_offset` reference the original text's character positions correctly, but the `chunk_text_str` content may not match `raw_text[start_offset:end_offset]` exactly due to the `.strip()` in `_split_sentences` and the space-join.
- **Impact:** Quote verification works against `chunk_text` (the joined string), not `raw_text`, so it's internally consistent. But if someone tries to verify `chunk_text == raw_text[start:end]`, it may not match.
- **Fix:** Use `text[start_offset:end_offset]` directly instead of joining.

#### m8 — No `updated_at` trigger at DB level
- **File:** [app/models/evidence.py](app/models/evidence.py#L31-L36), [app/models/clusters.py](app/models/clusters.py#L38-L40)
- **Problem:** `onupdate=func.now()` only works when the ORM processes the update. Direct SQL updates or bulk updates via `session.execute(update(...))` will NOT trigger the `onupdate` clause because it's ORM-level, not a DB trigger.
- **Impact:** `updated_at` may be stale after direct SQL operations.
- **Fix:** Add a Postgres `BEFORE UPDATE` trigger, or accept the limitation.

---

## 3. Design & Architecture Issues

### 🔴 CRITICAL

#### A1 — In-memory job store: all job state lost on server restart
- **File:** [app/routers/jobs.py](app/routers/jobs.py#L17)
- **Problem:** `_job_store: dict[UUID, dict] = {}` keeps all job state in process memory. A server restart (crash, deploy, docker restart) loses everything. Long-running extraction or embedding jobs that were in progress will have no record. Users polling for job status will get 404.
- **Impact:** Effectively makes the job system unreliable for production. Any crash during extraction loses the job's status permanently.
- **Fix:** Store jobs in a DB table (e.g., `jobs` table with status, timestamps, error, result_count). Redis is already provisioned.

#### A2 — In-memory LLM cost log: all observability data lost on restart
- **File:** [app/llm/client.py](app/llm/client.py#L57)
- **Problem:** `_call_log: list[LLMCallRecord] = []` and cost accumulator are purely in-memory. Restarting the server zeroes all cost tracking.
- **Impact:** No persistent record of LLM spend. Can't build dashboards, can't audit costs.
- **Fix:** Persist call records to the database or push to an external observability service.

### 🟠 MAJOR

#### A3 — Clustering runs synchronously but returns 202
- **File:** [app/routers/clusters.py](app/routers/clusters.py#L31-L38)
- **Problem:** `POST /clusters/run` runs `await run_threshold_clustering(session)` which blocks the request until all embeddings are processed. But it returns HTTP 202 ("Accepted — processing later"), which falsely implies async background processing. With many embeddings, this call can time out.
- **Impact:** Misleading API semantics. Potential request timeout on large datasets. The client may start polling for status when the work is already done.
- **Fix:** Either (a) change to 200 and return results immediately, or (b) move to a background task with job tracking like extraction/embedding.

#### A4 — Greedy clustering is order-dependent and non-deterministic
- **File:** [app/services/cluster_service.py](app/services/cluster_service.py#L50-L55)
- **Problem:** The query `select(ProblemEmbedding)` has no `ORDER BY`. The iteration order depends on database internals. Different runs on the same data can produce different clusters because which problem becomes a cluster seed depends on iteration order.
- **Impact:** Results are not reproducible. Testing and debugging are harder.
- **Fix:** Add `.order_by(ProblemEmbedding.created_at.asc())` or similar deterministic ordering.

#### A5 — No rate limiting or batching on LLM API calls
- **File:** [app/services/extraction_service.py](app/services/extraction_service.py#L17), [app/services/embeddings_service.py](app/services/embeddings_service.py#L15)
- **Problem:** Semaphore limits concurrency to 4, but there's no **rate limit** (requests per second) toward the Gemini API. With fast chunks, 4 concurrent requests can burst rapidly. Gemini API has rate limits (e.g., 60 RPM for free tier).
- **Impact:** API rate limit errors (429s from Google) that trigger retries, increasing latency and cost.
- **Fix:** Add a `TokenBucket` or `asyncio.Semaphore` combined with `asyncio.sleep()` to throttle to under Gemini's RPM limit.

#### A6 — No pagination on `GET /roadmap`
- **File:** [app/routers/clusters.py](app/routers/clusters.py#L98-L112), [app/services/cluster_service.py](app/services/cluster_service.py#L185-L200)
- **Problem:** `get_roadmap()` fetches ALL proposals with a JOIN. No `LIMIT` or `OFFSET`. As proposals grow, this becomes a full table scan returned in one response.
- **Impact:** Growing memory usage and response size.
- **Fix:** Add pagination params like other list endpoints.

#### A7 — No pagination on `GET /llm/calls`
- **File:** [app/routers/jobs.py](app/routers/jobs.py#L153-L156)
- **Problem:** Returns the entire `_call_log` array. After thousands of LLM calls, this becomes a huge JSON response.
- **Impact:** Slow response, high memory usage on serialization.
- **Fix:** Add pagination, or at minimum a `limit` parameter.

#### A8 — Proposal creation doesn't verify cluster has members
- **File:** [app/services/cluster_service.py](app/services/cluster_service.py#L148-L163)
- **Problem:** You can create a feature proposal for an empty cluster (mention_count=0) or one that has been orphaned. The roadmap would surface proposals with no backing evidence.
- **Impact:** Misleading roadmap — proposals with zero evidence support look equal to well-supported ones.
- **Fix:** Warn or reject proposals on empty clusters, or at minimum surface `mention_count=0` more prominently.

#### A9 — CORS hardcoded to localhost:3000 only
- **File:** [app/main.py](app/main.py#L31)
- **Problem:** CORS only allows `localhost:3000` and `127.0.0.1:3000`. Any deployment to a staging/production domain will break without code changes.
- **Impact:** Cannot deploy frontend to any non-localhost domain without editing source code.
- **Fix:** Make CORS origins configurable via env var (e.g., `CORS_ORIGINS`).

### 🟡 MINOR

#### A10 — Redis is provisioned but completely unused
- **File:** [docker-compose.yml](../docker-compose.yml#L36-L44), [app/config.py](app/config.py#L7)
- **Problem:** A Redis container runs in docker-compose, `redis_url` is in settings, and `redis>=5.0.0` is in requirements.txt. But nothing in the application code ever connects to Redis.
- **Impact:** Wasted resources (memory, a container). Confusing to new developers.
- **Recommendation:** Either use it (for job queue, caching, rate limiting), or remove it to reduce complexity.

#### A11 — `psycopg2-binary` in requirements.txt but asyncpg is the actual driver
- **File:** [requirements.txt](requirements.txt#L6)
- **Problem:** Both `asyncpg` and `psycopg2-binary` are in requirements. The app uses `asyncpg` exclusively. `psycopg2-binary` is never imported anywhere.
- **Impact:** Unnecessary dependency. Extra image size. Confusion about which driver is used.
- **Fix:** Remove `psycopg2-binary` unless Alembic needs it (check Alembic's `env.py`).

#### A12 — No health check for database connectivity
- **File:** [app/main.py](app/main.py#L60-L62)
- **Problem:** The health endpoint returns `{"status": "ok"}` unconditionally. It doesn't check if the database is reachable.
- **Impact:** Load balancers will consider the service healthy even if DB is down.
- **Fix:** Add a `SELECT 1` query inside the health check, return 503 if it fails.

---

## 4. Optimizations & Cleanup

### 🔵 Pointless/Wasteful

#### O1 — `version` field on `FeatureProposal` does nothing
- **File:** [app/models/clusters.py](app/models/clusters.py#L100)
- **Problem:** `version` column defaults to `1` and is never incremented, read, or used by any logic. No versioning system exists.
- **Impact:** False promise of functionality. Column wastes space.
- **Recommendation:** Either implement proposal versioning or remove the column.

#### O2 — `metadata_` JSONB columns on clusters and proposals are never used
- **File:** [app/models/clusters.py](app/models/clusters.py#L32-L34), [app/models/clusters.py](app/models/clusters.py#L101-L103)
- **Problem:** Both `ProblemCluster` and `FeatureProposal` have `metadata_` JSONB columns. No service reads them, no endpoint accepts them, no schema exposes them.
- **Impact:** Dead columns. Neither harmful nor useful.
- **Recommendation:** Remove from model/migration if no plan to use, or expose in schemas.

#### O3 — `extraction_job_id` on ProblemMention is never populated
- **File:** [app/models/problems.py](app/models/problems.py#L40)
- **Problem:** The column exists to link problems to their extraction job, but `_build_problem_mention()` never sets it, even though the `job_id` is available in the calling scope.
- **Impact:** No way to trace which job created which problems.
- **Fix:** Pass `job_id` through to `_build_problem_mention` and set it.

#### O4 — Duplicate `_call_with_retry` implementations
- **File:** [app/services/extraction_service.py](app/services/extraction_service.py#L151-L161), [app/services/embeddings_service.py](app/services/embeddings_service.py#L72-L82)
- **Problem:** Nearly identical retry-with-backoff functions exist in both files. Only difference is the log message string.
- **Impact:** Code duplication. Bug fix in one won't propagate to the other.
- **Fix:** Extract into a shared utility function (e.g., `app/utils/retry.py`).

#### O5 — Duplicate `ValueError` and generic `Exception` handlers in `_run_extract_job`
- **File:** [app/routers/jobs.py](app/routers/jobs.py#L101-L115)
- **Problem:** The `ValueError` except block and the generic `Exception` except block do the exact same thing (set status to "failed" with the error string). The `ValueError` catch is completely redundant.
- **Impact:** Unnecessary code. Confusing to readers.
- **Fix:** Remove the `ValueError` except block; the generic `Exception` already handles it.

#### O6 — `Iterable` import used but `Sequence`/`list` would be more accurate
- **File:** [app/services/extraction_service.py](app/services/extraction_service.py#L3), [app/services/embeddings_service.py](app/services/embeddings_service.py#L3)
- **Problem:** `from typing import Iterable` is imported and used to type the return of `.all()`, which actually returns a `list`. `Iterable` is too permissive and prevents methods like `len()`.
- **Impact:** Type annotation imprecision.
- **Fix:** Use `list[...]` directly, remove `Iterable` import.

### 🔵 Performance Optimizations

#### O7 — Clustering compares every embedding to every cluster centroid (O(n×k))
- **File:** [app/services/cluster_service.py](app/services/cluster_service.py#L56-L64)
- **Problem:** The inner loop iterates all existing clusters for each new embedding. With 10,000 embeddings and 500 clusters, that's 5 million numpy operations per run.
- **Impact:** Clustering time grows quadratically with data size.
- **Recommendation:** For now this is fine at small scale. When scaling, consider: (a) using a vector index for nearest-centroid lookup, (b) batch matrix operations with numpy, (c) switching to HDBSCAN which handles this natively.

#### O8 — `embed_text` calls are serialized per-problem despite `asyncio.to_thread`
- **File:** [app/services/embeddings_service.py](app/services/embeddings_service.py#L44-L52)
- **Problem:** `asyncio.to_thread(client.embed_text, text)` runs in a thread pool, but the Gemini SDK client is a single object. If the SDK is not thread-safe, concurrent calls could corrupt state. If it IS thread-safe, 4 concurrent `to_thread` calls create 4 OS threads for inherently I/O-bound HTTP calls.
- **Impact:** Either a thread-safety risk or suboptimal use of threads for network I/O.
- **Recommendation:** Investigate if the Gemini SDK is thread-safe. Consider using `aiohttp` for direct API calls to avoid the thread pool overhead.

#### O9 — `list_evidence` builds full ORM objects when only a few fields are needed
- **File:** [app/services/evidence_service.py](app/services/evidence_service.py#L81-L87)
- **Problem:** The query selects full `Evidence` ORM objects (including `raw_text` which can be very large), then manually picks a few fields to build dicts. The `raw_text` is loaded but never used in list responses.
- **Impact:** Unnecessary memory and I/O for large documents. With 100 evidence documents of 50KB each, that's 5MB loaded just for a list page.
- **Recommendation:** Use `select(Evidence.id, Evidence.title, ...)` to select only needed columns, or use `defer(Evidence.raw_text)` to lazy-load the text column.

#### O10 — Rate limiter bucket is never pruned for disappeared IPs
- **File:** [app/middleware/rate_limit.py](app/middleware/rate_limit.py#L31)
- **Problem:** `self._buckets` is a `defaultdict(list)` that only cleans timestamps within a bucket when that IP makes a new request. IPs that stop making requests leave stale entries forever.
- **Impact:** Slow memory leak over time. Millions of unique IPs = millions of dict entries.
- **Recommendation:** Add periodic cleanup or use a TTL cache (e.g., `cachetools.TTLCache`).

#### O11 — Alembic migration adds vector column via raw SQL, ORM model defines it via Vector()
- **File:** [alembic/versions/001_phase1_tables.py](alembic/versions/001_phase1_tables.py#L90-L91) vs [app/models/embeddings.py](app/models/embeddings.py#L27)
- **Problem:** The migration uses `op.execute("ALTER TABLE ... ADD COLUMN embedding vector(768)")` while the ORM model uses `Vector(768)` from `pgvector.sqlalchemy`. This works fine, but if `create_all` runs (as it does in `startup()`), it will try to create the table again. Since `create_all` uses `checkfirst=True` by default, it skips existing tables — but this split between migration-managed and ORM-managed schemas is fragile.
- **Impact:** Potential for drift between migration and ORM definitions. The `startup()` `create_all` is redundant with Alembic.
- **Recommendation:** Pick one source of truth: either rely solely on Alembic and remove `create_all`, or remove Alembic and let `create_all` handle everything.

# Jan 16 Tasks (Backend Strategy Execution)

This list mirrors the strategy-backed gaps and puts them into executable tasks with details, common mistakes, and optimization insights.

1. Add `docker-compose.yml` for API + Postgres (pgvector) + Redis
   Description: Define local dev stack with `api`, `db` (pgvector/pg16), `redis`, and optionally `worker` later. Use the environment variables in `backend/.env` and expose ports `8000`, `5432`, `6379`.
   Common mistakes: Forgetting pgvector image (using vanilla Postgres), missing volume for db persistence, incorrect `DATABASE_URL` scheme (`postgresql+asyncpg://`), or forgetting to align the compose `service` names with connection strings.
   Optimization insights: Use `healthcheck` for db/redis and `depends_on` with health conditions; mount a volume for database; keep compose minimal until Celery is added.

2. Create database schema/migrations for Phase 1 tables
   Description: Add Alembic (or equivalent) and create migrations for `evidence`, `evidence_chunks`, `problem_mentions`, and `problem_embeddings`. Ensure pgvector extension is enabled and indexes match the strategy (including GIN for tags and ivfflat for vector).
   Common mistakes: Skipping `vector` extension migration, missing indexes, wrong vector dimension (should match embedding model, currently 768), not setting `ondelete="CASCADE"` correctly, or mismatching table names vs models.
   Optimization insights: Keep migrations idempotent; add `create extension if not exists vector`; use `ivfflat` index after some data volume is present for better performance.

3. Implement similarity search endpoint for problems
   Description: Add `GET /api/v1/problems/similar?text=...` that embeds the query and performs kNN search against `problem_embeddings`, joining back to `problem_mentions` for results.
   Common mistakes: Forgetting to normalize/query text, returning embeddings instead of problem data, not limiting results, or ignoring `pgvector` distance functions and indexes.
   Optimization insights: Add `limit` and `min_score` parameters; use cosine distance and return score; cache embeddings for repeated queries.

4. Add quote verification after extraction
   Description: After LLM extraction, verify each `quote_text` exists in the source chunk text. If not found, either drop the mention or attempt fuzzy matching before storing.
   Common mistakes: Blindly trusting `quote_text`, storing invalid offsets, or failing when quotes appear multiple times in a chunk.
   Optimization insights: Store `quote_start`/`quote_end` reliably; add fuzzy matching with a threshold; log verification failures to improve prompts.

5. Add problems stats endpoint (optional Phase 1)
   Description: Implement `GET /api/v1/problems/stats` for aggregate counts by persona, severity, and tag to satisfy the API map.
   Common mistakes: Expensive aggregate queries without indexes, or returning inconsistent filter semantics.
   Optimization insights: Use indexed columns (`severity`, `persona`, `tags`), and allow date filtering to keep queries fast.

6. Phase 2 planning: clustering, proposals, citations, roadmap
   Description: Scaffold the models, schemas, and services for clusters and feature proposals, plus the prioritization engine and roadmap endpoint.
   Common mistakes: Building clustering before embeddings are stable, skipping provenance links from proposals to problem mentions, or ignoring prompt versioning.
   Optimization insights: Start with threshold clustering, keep cluster summaries cached, and store proposal versions for auditability.

7. Phase 3 planning: task trees, job queue, cost tracking, prompt versioning, eval harness
   Description: Introduce Celery + Redis for async jobs, store cost metrics per LLM call, and add a golden-set eval harness for extraction regression.
   Common mistakes: Mixing in-memory job status with Celery, no retry strategy, or no prompt version metadata.
   Optimization insights: Centralize LLM call wrappers with tracing, keep cost and prompt metadata in one place, and run evals in CI.

8. Phase 4 planning: auth, rate limiting, DB optimization, production Docker/monitoring
   Description: Add API key auth with a migration path to JWT, rate limiting middleware, DB tuning, and production-grade Docker config.
   Common mistakes: Retrofitting auth after public endpoints are already used, or missing request validation and rate limiting on LLM-heavy routes.
   Optimization insights: Add basic API key guardrails early; use connection pooling; use structured logs for cost and error monitoring.

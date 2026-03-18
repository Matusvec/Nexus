# Nexus Architecture

> Evidence-driven problem discovery and prioritization platform.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Data Model](#3-data-model)
4. [Pipeline Flow](#4pipeline-flow)
5. [Backend Architecture](#5-backend-architecture)
6. [Frontend Architecture](#6-frontend-architecture)
7. [API Endpoints](#7-api-endpoints)
8. [Pages & Routes](#8-pages--routes)
9. [Frontend Revamp Status](#9-frontend-reamp-status)
10. [Future Roadmap](#10future-roadmap)

---

## 1. Project Overview

**Nexus PM** is an evidence-driven problem discovery and prioritization platform. Users upload raw customer signals (interviews, support tickets, sales notes, surveys), and the system:

1. **Chunks** the text into processable segments
2. **Extracts** structured problem mentions using LLM (Gemini 2.0 Flash)
3. **Embeds** problems as vectors for similarity search
4. **Clusters** related problems into pain themes
5. **Generates** feature proposals with citations back to source quotes
6. **Breaks down** proposals into dev-ready task trees
7. **Ranks** everything into a prioritized roadmap

Every insight traces back to direct customer quote. No hand-waving.

---

## 2. Tech Stack

| Layer | Technology |
|----|-----------|
| **Frontend** | Next.js 16, TypeScript, Tailwind CSS, shadcn/ui |
| **Backend** | FastAPI (Python), async, Pydantic validation |
| **Database** | PostgreSQL 16 + pgvector (structured data + vector embeddings) |
| **LLM** | Gemini 2.0 Flash (extraction + proposals) + text-embedding-004 (embeddings) |
| **Cache** | Redis 7 (provisioned for rate limiting / future use) |
| **Containers** | Docker Compose |

---

## 3. Data Model

```
evidence (1) ──────< evidence_chunks (N)
                         │
                         ▼  (LLM extracts)
                    problem_mentions (N)
                         │
                         ├──── problem_embeddings (1:1)
                         │
                         ├────> cluster_memberships (N) >──── problem_clusters (1)
                         │                                        │
                         │                                        ▼  (LLM generates)
                         │                                        feature_proposals (N)
                         └──────────────< proposal_citations (N) ─────┘
```

---

## 4. Pipeline Flow

```
Evidence Upload → Chunking → Problem Extraction (LLM) → Embedding → Clustering
                                                                        │
                                                                        ▼
                              Prioritized Roadmap ← Scoring ← Feature proposals (LLM)
                                                                        │
                                                                        ▼
                                                                   Task Trees (LLM)
```

---

## 5. Backend Architecture

### Stack & Infrastructure

| Component | Technology | Details |
|--------|--------|---------|
| API Framework | FastAPI | Async-native, Pydantic validation, auto OpenAPI docs |
| Database | PostgreSQL + pgvector | Structured data + 768-dim vector embeddings in one DB |
| ORM | SQLAlchemy 2.0 (async) | AsyncSession via asyncpg driver |
| Migrations | Alembic | Phase 1 + Phase 2 tables |
| LLM Provider | Gemini 2.0 Flash | JSON generation + text-embedding-004 for vectors |
| Job Execution | FastAPI BackgroundTasks | In-memory, no Celery/Redis queue yet |
| Containerization | Docker Compose | API + Postgres (pgvector:pg16) + Redis containers |
| Rate Limiting | In-memory sliding window log | Per-IP, configurable via env vars |
| Auth | API key middleware | Built but not wired to any route (dev mode) |

### Docker Compose Services

```
api      →  FastAPI on port 8000
db       →  pgvector/pgvector:pg16 on port 5432
redis    →  redis:7-alpine on port 6379 (provisioned but not used by app yet)
```

### Entry Point

**File:** `backend/app/main.py`

The FastAPI app is created with two middleware layers (rate limiting, CORS) and four routers. On startup:

1. `CREATE EXTENSION IF NOT EXISTS vector` — ensures pgvector available
2. `Base.metadata.create_all` — auto-creates all SQLAlchemy-defined tables
3. All 8 ORM models imported to register with the metadata

### Configuration

**File:** `backend/app/config.py`

Uses Pydantic `BaseSettings` to load from `.env`:

| Variable | Default | Purpose |
|--------|--------|--------|
| `DATABASE_URL` | required | Postgres connection string |
| `REDIS_URL** | redis://localhost:6379/0** | Reserved for future Celery worker |
| `GEMINI_API_KEY** | required** | Google AI API key |
| `GEMINI_MODEL** | gemini-2.0-flash** | Model for JSON generation |
| `GEMINI_EMBEDDING_MODEL** | text-embedding-004** | Model for embeddings (768 dims) |
| `CHUNK_MAX_TOKENS** | 500** | Max tokens per evidence chunk |
| `CHUNK_OVERLAP_TOKENS** | 50** | Overlap between chunks |
| `RATE_LIMIT_REQUESTS** | 60** | Max requests per window per IP |
| `RATE_LIMIT_WINDOW** | 60** | Window size in seconds |

---

## 6. Frontend Architecture

### Tech Stack

| Component | Technology |
|--------|--------|
| Framework | Next.js 16 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS + shadcn/ui |
| Icons | Lucide React |
| Font | Fraunces (display), IBM Plex Sans (body), IBM Plex Mono (code |

### Design Philosophy

Nexus PM should feel like **trusted analyst** — confident enough to surface hard truths, intelligent enough to find signal, fast enough to never block, polished enough to trust every pixel.

| Attribute | Expression |
|--------|--------|
| **Confident** | Bold typography hierarchy, decisive color usage, no wishy-washy "maybe" UI |
| **Intelligent** | Data density without clutter, severity distributions, priority scores, citation links |
| **Fast** | Instant navigation via client-side transitions, skeleton states instead of blank screens |
| **Premium** | Micro-shadows on cards, consistent 4px radius increments, system-level font rendering |

### UX Principles

1. **Clarity over clutter** — Show one clear hierarchy per screen. If user can't identify primary action within 2 seconds, page needs redesign.
2. **Guided interaction** — Every page implies next step. Evidence → "Extract problems." cluster → "Generate proposal." proposal → "Approve & generate tasks." pipeline is UX.
3. **Minimal friction** — Upload should <3 clicks. Navigation should never exceed 2 levels deep. filters persist across sessions. back buttons always work.
4. **Evidence is king** — Every data point traces back to verbatim quote. every quote traces to source document. citation links are first-class citizens, not footnotes.
5. **Async-aware, never blocked** — LLM operations take 5-30 seconds. UI shows progress bars with estimated completion, toast notifications on finish, lets users navigate freely while jobs run.
6. **Density over decoration** — PMs scan 50+ rows of data. tables are primary pattern. cards are for overview screens. white space serves readability, not aesthetics.

### Color Strategy

Nexus PM uses **warm-neutral light theme** with strategic yellow and blue accents. yellow represents **insight and energy** (product output). blue represents **trust and depth** (product intelligence). They used sparingly — as signals, not wallpaper.

#### Primary Palette

| Token | Hex | Usage |
|--------|--------|--------|
| `--nexus-blue** | #0E7490** | Primary CTAs, active sidebar items, links, pipeline "complete" indicators |
| `--nexus-yellow** | #E88C0A** | Accent highlights, "running" status, notification badges, score accents |
| `--nexus-amber** | #F59E0B** | Secondary accent, hover state lift on yellow elements |

#### Neutral Palette

| Token | hex | Usage |
|--------|--------|--------|
| `--surface-0** | #FAFAF6** | Page background (warm off-white, avoids clinical white) |
| `--surface-1** | #F5F3EE** | Card backgrounds, sidebar bg |
| `--surface-2** | #EBE8E0** | Input backgrounds, muted section fills |
| `--surface-3** | #DDD9CE** | Borders, dividers |
| `--ink-primary** | #1A2332** | Headings, primary text |
| `--ink-secondary** | #4A5568** | Body text, descriptions |
| `--ink-muted** | #8A9AB5** | Captions, timestamps, placeholders |

#### Semantic Colors

| Token | Hex | Usage |
|--------|--------|--------|
| `--severity-critical** | #DC2626** | Critical severity badges, destructive actions |
| `--severity-high** | #EA580C** | High severity |
| `--severity-medium** | #D97706** | Medium severity |
| `--severity-low** | #16A34A** | Low severity, success states |
| `--status-draft** | #3B82F6** | Draft proposals |
| `--status-approved** | #16A34A** | Approved proposals |
| `--status-rejected** | #9CA3AF** | Rejected proposals |
| `--status-running** | #E88C0A** | Active jobs, processing |
| `--status-error** | #DC2626** | Failed jobs, errors |

### Typography

| Role | Font | Usage |
|--------|--------|--------|
| **Display** (h1, h2) | Fraunces (variable, optical size) | 600–700 | headings |
| **Body / UI** | IBM Plex Sans | 400, 500, 600 | descriptions, UI elements |
| **Code / Monospace** | IBM Plex Mono | 400 | technical contexts |

### Type Scale

| Level | font | Size | weight | Usage |
|--------|--------|--------|--------|--------|
| **Page Title** | Fraunces | 30px | 600 | one per page |
| **Section Head** | Fraunces | 22px | 600 | card group titles |
| **Card Title** | IBM Plex Sans | 16px | 600 | evidence title in table |
| **Body** | IBM Plex Sans | 14px | 400 | descriptions |
| **Caption** | IBM Plex Sans | 12px | 400 | timestamps |

---

## 7. API Endpoints

| Method | Endpoint | Description |
|--------|--------|--------|
| POST | /api/v1/evidence | Upload new evidence |
| GET | /api/v1/evidence | List evidence (paginated) |
| GET | /api/v1/evidence/{id} | Evidence detail with chunks |
| POST | /api/v1/evidence/{id}/extract | Trigger problem extraction |
| GET | /api/v1/problems | List problems (filterable) |
| GET | /api/v1 problems/{id}/similar | Find similar problems |
| POST | /api/v1/clusters/run | Run clustering algorithm |
| GET | /api/v1/clusters | List clusters |
| POST | /api/v1 clusters/{id}/propose | Generate feature proposal |
| GET | /api/v1/roadmap | Get prioritized roadmap |
| GET | /api/v1/jobs/{id} | Check async job status |

Full API spec: [frontend/API_SPECIFICATION.md](frontend/API_SPECIFICATION.md)

---

## 8. Pages & Routes

| Route | Description |
|--------|--------|
| /pm | Dashboard — pipeline overview with counts and status |
| /pm/evidence | Browse and upload source material |
| /pm/evidence/upload | Upload new evidence with metadata |
| /pm/problems | Filterable table all extracted problem mentions |
| /pm/clusters | Grouped pain themes with severity breakdowns |
| /pm/proposals | AI-generated feature proposals with citations |
| /pm/proposals/{id}/tasks | Dev-ready task trees per proposal |
| /pm/roadmap | Prioritized ranking of proposals |
| /pm/settings | Configuration (API keys, prompt versions) |
| /pm/usage | LLM cost tracking and job history |

---

## 9. Frontend Revamp Status

STATUS: **IN PROGRESS**

### What's Being Updated

- [ ] Architecture.md created
- [ ] Frontend design improvements
- [ ] Color system polish
- [ ] Typography updates
- [ ] Animations & interactions
- [ ] Page flows revamp
- [ ] Empty states premium feel

### What To Do

1. Create/update design system globals.css
2. Polish page header component styles
3. Add animations to transitions effects
4. Improve sidebar navigation premium feel
5. Improve empty states better styling
6. Add more page flows polish
7. Make cards look premium
8. Make tables look more professional

---

## 10. Future Roadmap

Phase 1:
- [ ] docker-compose.yml for API + Postgres (pgvector) + Redis
- [ ] database schema/migrations for Phase 1 tables
- [ ] similarity search endpoint for problems
- [ ] quote verification after extraction
- [ ] problems stats endpoint

Phase 2:
- [ ] clustering, proposals, citations, roadmap

Phase 3:
- [ ] task trees, job queue, cost tracking, prompt versioning, eval harness

Phase 4:
- [ ] auth, rate limiting, DB optimization, production Docker/monitoring

---

*Last updated: Wed 2026-03-18 17:04 UTC*
*This document updates as we go — refer back to it for current status.
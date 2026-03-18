# Nexus PM

**Cursor for Product Managers** — Transform messy customer evidence into roadmap-grade decisions and dev-ready task breakdowns, all traceable back to direct quotes.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+pgvector-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

---

## What It Does

Nexus PM is an evidence-driven problem discovery and prioritization platform. You upload raw customer signals (interviews, support tickets, sales notes, surveys), and the system automatically:

1. **Chunks** the text into processable segments
2. **Extracts** structured problem mentions using LLM (Gemini 2.0 Flash)
3. **Embeds** problems as vectors for similarity search
4. **Clusters** related problems into pain themes (UMAP + HDBSCAN)
5. **Generates** feature proposals with citations back to source quotes
6. **Breaks down** proposals into dev-ready task trees
7. **Ranks** everything into a prioritized roadmap

Every insight traces back to a direct customer quote. No hand-waving.

---

## Pipeline

```
Evidence Upload → Chunking → Problem Extraction (LLM) → Embedding → Clustering
                                                                        │
                                                                        ▼
                              Prioritized Roadmap ← Scoring ← Feature Proposals (LLM)
                                                                        │
                                                                        ▼
                                                                   Task Trees (LLM)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui |
| **Backend** | FastAPI (Python 3.12), async SQLAlchemy, Pydantic v2 |
| **Database** | PostgreSQL 16 + pgvector (structured data + vector embeddings in one DB) |
| **LLM** | Gemini 2.0 Flash (extraction + proposals) + text-embedding-004 (768-dim embeddings) |
| **ML** | UMAP + HDBSCAN for semantic clustering, scikit-learn |
| **Task Queue** | Celery + Redis (async job processing) |
| **Containers** | Docker Compose (5 services) |

---

## Pages

| Route | Description |
|-------|-------------|
| `/pm` | Dashboard — pipeline overview with counts and next actions |
| `/pm/evidence` | Browse and manage source material (interviews, tickets, etc.) |
| `/pm/evidence/upload` | Upload new evidence with metadata |
| `/pm/problems` | Filterable table of all extracted problem mentions |
| `/pm/clusters` | Grouped pain themes with severity breakdowns |
| `/pm/proposals` | AI-generated feature proposals with status filtering |
| `/pm/proposals/[id]` | Proposal detail with citations, risks, and metrics |
| `/pm/proposals/[id]/tasks` | Dev-ready task tree visualization |
| `/pm/roadmap` | Prioritized ranking of proposals |
| `/pm/usage` | LLM cost tracking and job history |

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose)
- A [Gemini API key](https://aistudio.google.com/apikey)

### Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/Matusvec/Nexus.git
   cd Nexus
   ```

2. **Create backend env file**
   ```bash
   cp backend/.env.example backend/.env
   ```
   Edit `backend/.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_key_here
   ```

3. **Start everything**
   ```bash
   docker compose up --build
   ```

4. **Open the app**
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## Project Structure

```
Nexus/
├── docker-compose.yml          # Orchestrates all 5 services
├── frontend/                   # Next.js app
│   ├── app/pm/                 # All PM pipeline pages
│   ├── components/pm/          # PM-specific components
│   ├── components/ui/          # shadcn/ui components
│   └── lib/pm/                 # API client, types, utilities
├── backend/                    # FastAPI app
│   ├── app/
│   │   ├── main.py             # Entry point, routers, startup
│   │   ├── config.py           # Environment configuration
│   │   ├── database.py         # Async SQLAlchemy + connection pool
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── schemas/            # Pydantic request/response schemas
│   │   ├── routers/            # API endpoints
│   │   ├── services/           # Business logic (extraction, clustering, embeddings)
│   │   ├── llm/                # Multi-provider LLM wrapper
│   │   ├── middleware/         # Auth (API key), rate limiting
│   │   └── utils/              # Chunking, retry logic
│   ├── prompts/                # Versioned LLM prompt templates
│   └── alembic/                # Database migrations
```

---

## Data Model

```
evidence (1) ──────< evidence_chunks (N)
                          │
                          ▼  (LLM extracts)
                    problem_mentions (N)
                          │
                          ├──── problem_embeddings (1:1)
                          │
                          ├────> cluster_memberships (N) >──── problem_clusters (1)
                          │                                         │
                          │                                         ▼  (LLM generates)
                          │                                  feature_proposals (N)
                          │                                         │
                          └──────────────< proposal_citations (N) ──┘
                                                                    │
                                                                    ▼  (LLM generates)
                                                              task_items (N)
```

---

## API Endpoints

### Evidence
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/evidence` | Upload new evidence |
| `GET` | `/api/v1/evidence` | List evidence (paginated) |
| `GET` | `/api/v1/evidence/{id}` | Evidence detail with chunks |
| `PUT` | `/api/v1/evidence/{id}` | Update evidence |
| `DELETE` | `/api/v1/evidence/{id}` | Delete evidence |

### Problems
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/problems` | List problems (filterable by severity, persona, tag) |
| `GET` | `/api/v1/problems/{id}` | Problem detail |
| `GET` | `/api/v1/problems/similar` | Semantic similarity search |
| `GET` | `/api/v1/problems/stats` | Aggregated statistics |

### Clusters & Proposals
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/clusters/run` | Run threshold-based clustering |
| `POST` | `/api/v1/clusters/run_hdbscan` | Run HDBSCAN clustering |
| `GET` | `/api/v1/clusters` | List clusters (paginated) |
| `GET` | `/api/v1/clusters/{id}` | Cluster detail with members |
| `POST` | `/api/v1/clusters/{id}/summarize` | Generate cluster summary |
| `POST` | `/api/v1/clusters/{id}/generate_proposal` | Generate feature proposal |
| `GET` | `/api/v1/proposals` | List proposals (filterable by status) |
| `POST` | `/api/v1/proposals` | Create proposal |
| `GET` | `/api/v1/proposals/{id}` | Proposal detail with citations |
| `PUT` | `/api/v1/proposals/{id}` | Update proposal |
| `DELETE` | `/api/v1/proposals/{id}` | Delete proposal |
| `POST` | `/api/v1/proposals/{id}/approve` | Approve proposal |
| `POST` | `/api/v1/proposals/{id}/reject` | Reject proposal |
| `POST` | `/api/v1/proposals/{id}/regenerate` | Regenerate with LLM |

### Tasks & Roadmap
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/proposals/{id}/tasks` | Task tree for a proposal |
| `PATCH` | `/api/v1/tasks/{id}` | Update task |
| `GET` | `/api/v1/roadmap` | Prioritized roadmap |
| `POST` | `/api/v1/roadmap/score` | Score and rank proposals |
| `PATCH` | `/api/v1/roadmap/{id}/weight` | Adjust strategic weight |

### Jobs & Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/jobs/extract_problems` | Trigger extraction job |
| `POST` | `/api/v1/jobs/embed_problems` | Trigger embedding job |
| `POST` | `/api/v1/jobs/generate_tasks` | Trigger task generation |
| `POST` | `/api/v1/jobs/generate_proposal` | Trigger proposal generation |
| `GET` | `/api/v1/jobs/{id}/status` | Check job status |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/metrics` | Operational metrics |
| `GET` | `/api/v1/llm/costs` | LLM cost summary |
| `GET` | `/api/v1/llm/calls` | LLM call history |

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | required | PostgreSQL connection string |
| `GEMINI_API_KEY` | required | Google AI API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | LLM for extraction/generation |
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Embedding model (768 dims) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection for Celery |
| `CHUNK_MAX_TOKENS` | `500` | Max tokens per evidence chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between chunks |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window per IP |
| `RATE_LIMIT_WINDOW` | `60` | Window size in seconds |

---

## Architecture Docs

- [Backend Architecture](backend/ARCHITECTURE.md) — Full technical reference
- [Backend Strategy](backend/strategybackend.md) — Design decisions and approach
- [Frontend Strategy](frontend/strategyfrontend.md) — UI flows, wireframes, component architecture

---

## License

[MIT](LICENSE)

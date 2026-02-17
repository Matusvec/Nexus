# Nexus PM

**Cursor for Product Managers** — Transform messy customer evidence into roadmap-grade decisions and dev-ready task breakdowns, all traceable to quotes.

---

## What It Does

Nexus PM is an evidence-driven problem discovery and prioritization platform. You upload raw customer signals (interviews, support tickets, sales notes, surveys), and the system automatically:

1. **Chunks** the text into processable segments
2. **Extracts** structured problem mentions using LLM (Gemini 2.0 Flash)
3. **Embeds** problems as vectors for similarity search
4. **Clusters** related problems into pain themes
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
| **Frontend** | Next.js 16, TypeScript, Tailwind CSS, shadcn/ui |
| **Backend** | FastAPI (Python), async, Pydantic validation |
| **Database** | PostgreSQL 16 + pgvector (structured data + vector embeddings in one DB) |
| **LLM** | Gemini 2.0 Flash (extraction + proposals) + text-embedding-004 (embeddings) |
| **Cache** | Redis 7 (provisioned for rate limiting / future use) |
| **Containers** | Docker Compose |

---

## Pages

| Route | Description |
|-------|-------------|
| `/pm` | Dashboard — pipeline overview with counts and status |
| `/pm/evidence` | Browse and upload source material (interviews, tickets, etc.) |
| `/pm/evidence/upload` | Upload new evidence with metadata |
| `/pm/problems` | Filterable table of all extracted problem mentions |
| `/pm/clusters` | Grouped pain themes with severity breakdowns |
| `/pm/proposals` | AI-generated feature proposals with citations |
| `/pm/proposals/[id]/tasks` | Dev-ready task trees per proposal |
| `/pm/roadmap` | Prioritized ranking of proposals |
| `/pm/settings` | Configuration (API keys, prompt versions) |
| `/pm/usage` | LLM cost tracking and job history |

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- A [Gemini API key](https://aistudio.google.com/apikey)

### Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/newNexus.git
   cd newNexus
   ```

2. **Create backend env file**
   ```bash
   cp backend/.env.example backend/.env
   ```
   Add your Gemini API key:
   ```
   GEMINI_API_KEY=your_key_here
   DATABASE_URL=postgresql+asyncpg://nexus:nexus@db:5432/nexus
   REDIS_URL=redis://redis:6379/0
   ```

3. **Start everything**
   ```bash
   docker compose up --build
   ```

4. **Open the app**
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Project Structure

```
newNexus/
├── docker-compose.yml          # Orchestrates all 4 services
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
│   │   ├── routers/            # API endpoints (evidence, problems, clusters, jobs)
│   │   ├── services/           # Business logic (extraction, clustering, embeddings)
│   │   ├── llm/                # Gemini client wrapper
│   │   ├── middleware/         # Auth, rate limiting
│   │   └── utils/              # Chunking, retry logic
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
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/evidence` | Upload new evidence |
| `GET` | `/api/v1/evidence` | List evidence (paginated) |
| `GET` | `/api/v1/evidence/{id}` | Evidence detail with chunks |
| `POST` | `/api/v1/evidence/{id}/extract` | Trigger problem extraction |
| `GET` | `/api/v1/problems` | List problems (filterable) |
| `GET` | `/api/v1/problems/{id}/similar` | Find similar problems |
| `POST` | `/api/v1/clusters/run` | Run clustering algorithm |
| `GET` | `/api/v1/clusters` | List clusters |
| `POST` | `/api/v1/clusters/{id}/propose` | Generate feature proposal |
| `GET` | `/api/v1/roadmap` | Get prioritized roadmap |
| `GET` | `/api/v1/jobs/{id}` | Check async job status |

Full API specification: [`frontend/API_SPECIFICATION.md`](frontend/API_SPECIFICATION.md)

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | required | PostgreSQL connection string |
| `GEMINI_API_KEY` | required | Google AI API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | LLM for extraction/generation |
| `GEMINI_EMBEDDING_MODEL` | `text-embedding-004` | Embedding model (768 dims) |
| `CHUNK_MAX_TOKENS` | `500` | Max tokens per evidence chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between chunks |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window per IP |
| `RATE_LIMIT_WINDOW` | `60` | Window size in seconds |

---

## Architecture Docs

- [Backend Architecture](backend/ARCHITECTURE.md) — Full technical reference of everything built
- [Backend Strategy](backend/strategybackend.md) — Design decisions and roadmap
- [Frontend Strategy](frontend/strategyfrontend.md) — UI flows, wireframes, component architecture
- [API Specification](frontend/API_SPECIFICATION.md) — Endpoint contracts between frontend and backend

---

## License

[MIT](LICENSE)

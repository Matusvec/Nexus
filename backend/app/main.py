import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import engine, get_session
from app.logging_config import setup_logging
from app.middleware.rate_limit import RateLimitMiddleware
from app.routers import evidence, jobs, problems
from app.routers import clusters as clusters_router
from app.routers import tasks as tasks_router

# Initialize structured logging
setup_logging()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _get_cors_origins() -> list[str]:
    """Return CORS origins from env var, falling back to localhost defaults."""
    if settings.cors_origins:
        return [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure pgvector extension exists.

    Note: Table creation is managed by Alembic migrations.
    ``Base.metadata.create_all`` is intentionally NOT called here (O11)
    to avoid dual-source schema drift.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    logger.info("Database ready (pgvector extension verified)")
    yield


app = FastAPI(title="Nexus Backend", lifespan=lifespan)

# Rate limiting (configurable via env)
app.add_middleware(
    RateLimitMiddleware,
    rate=settings.rate_limit_requests,
    window=settings.rate_limit_window,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.middleware.auth import verify_api_key

auth_dep = [Depends(verify_api_key)]
app.include_router(evidence.router, prefix="/api/v1", tags=["evidence"], dependencies=auth_dep)
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"], dependencies=auth_dep)
app.include_router(problems.router, prefix="/api/v1", tags=["problems"], dependencies=auth_dep)
app.include_router(clusters_router.router, prefix="/api/v1", tags=["clusters"], dependencies=auth_dep)
app.include_router(tasks_router.router, prefix="/api/v1", tags=["tasks"], dependencies=auth_dep)


@app.get("/api/v1/health")
async def health_check() -> dict:
    """Health check with database connectivity verification."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "detail": "Database unreachable"},
        )


@app.get("/api/v1/metrics")
async def metrics_endpoint(session: AsyncSession = Depends(get_session)) -> dict:
    """Return operational stats for the Nexus backend."""
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

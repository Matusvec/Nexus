import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import settings
from app.database import engine
from app.middleware.rate_limit import RateLimitMiddleware
from app.routers import evidence, jobs, problems
from app.routers import clusters as clusters_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nexus Backend")

# Rate limiting (configurable via env)
app.add_middleware(
    RateLimitMiddleware,
    rate=settings.rate_limit_requests,
    window=settings.rate_limit_window,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    """Ensure pgvector extension exists and tables are created."""
    from app.database import Base  # noqa: F811
    from app.models import (  # noqa: F401
        Evidence, EvidenceChunk, ProblemMention, ProblemEmbedding,
        ProblemCluster, ClusterMembership, FeatureProposal, ProposalCitation,
    )

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified / created")


app.include_router(evidence.router, prefix="/api/v1", tags=["evidence"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
app.include_router(problems.router, prefix="/api/v1", tags=["problems"])
app.include_router(clusters_router.router, prefix="/api/v1", tags=["clusters"])


@app.get("/api/v1/health")
async def health_check() -> dict:
    return {"status": "ok"}

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import settings
from app.database import engine
from app.middleware.rate_limit import RateLimitMiddleware
from app.routers import evidence, jobs, problems
from app.routers import clusters as clusters_router
from app.routers import tasks as tasks_router

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

app.include_router(evidence.router, prefix="/api/v1", tags=["evidence"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
app.include_router(problems.router, prefix="/api/v1", tags=["problems"])
app.include_router(clusters_router.router, prefix="/api/v1", tags=["clusters"])
app.include_router(tasks_router.router, prefix="/api/v1", tags=["tasks"])


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

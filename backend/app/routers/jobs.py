import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.database import AsyncSessionLocal
from app.llm.client import get_call_log, get_cost_summary
from app.schemas.embeddings import EmbedProblemsRequest
from app.schemas.jobs import JobResponse, JobStatusResponse
from app.schemas.problems import ExtractProblemsRequest
from app.services.embeddings_service import embed_problems
from app.services.extraction_service import extract_problems_for_evidence

router = APIRouter()

_job_store: dict[UUID, dict] = {}
_job_lock = asyncio.Lock()


@router.post(
    "/jobs/extract_problems",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def extract_problems_endpoint(
    payload: ExtractProblemsRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    job_id = uuid4()
    now = datetime.now(timezone.utc)
    async with _job_lock:
        _job_store[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "job_type": "extract_problems",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result_count": None,
        }

    background_tasks.add_task(
        _run_extract_job, job_id, payload.evidence_id, payload.max_chunks
    )
    return JobResponse(job_id=job_id, status="pending")


@router.post(
    "/jobs/embed_problems",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def embed_problems_endpoint(
    payload: EmbedProblemsRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    job_id = uuid4()
    now = datetime.now(timezone.utc)
    async with _job_lock:
        _job_store[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "job_type": "embed_problems",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result_count": None,
        }

    background_tasks.add_task(_run_embed_job, job_id, payload.limit)
    return JobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: UUID) -> JobStatusResponse:
    async with _job_lock:
        job = _job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


async def _run_extract_job(
    job_id: UUID, evidence_id: UUID, max_chunks: int | None
) -> None:
    await _update_job(job_id, status="running", started_at=datetime.now(timezone.utc))
    try:
        async with AsyncSessionLocal() as session:
            mentions = await extract_problems_for_evidence(
                session, evidence_id, max_chunks
            )
        await _update_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc),
            result_count=len(mentions),
        )
    except ValueError as exc:
        await _update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc),
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        await _update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc),
            error=str(exc),
        )


async def _run_embed_job(job_id: UUID, limit: int | None) -> None:
    await _update_job(job_id, status="running", started_at=datetime.now(timezone.utc))
    try:
        async with AsyncSessionLocal() as session:
            embedded_ids = await embed_problems(session, limit)
        await _update_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc),
            result_count=len(embedded_ids),
        )
    except Exception as exc:  # noqa: BLE001
        await _update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc),
            error=str(exc),
        )


async def _update_job(job_id: UUID, **updates: object) -> None:
    async with _job_lock:
        job = _job_store.get(job_id)
        if not job:
            return
        job.update(updates)


# ── Cost tracking endpoints ─────────────────────────────────────

@router.get("/llm/costs")
async def llm_cost_summary() -> dict:
    """Return aggregate LLM cost metrics."""
    return get_cost_summary()


@router.get("/llm/calls")
async def llm_call_log() -> list[dict]:
    """Return detailed log of all LLM calls in this process lifetime."""
    return get_call_log()

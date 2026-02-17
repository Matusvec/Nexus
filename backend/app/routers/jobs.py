import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal, get_session
from app.models.jobs import Job, LLMCallLog
from app.schemas.embeddings import EmbedProblemsRequest
from app.schemas.jobs import JobResponse, JobStatusResponse
from app.schemas.problems import ExtractProblemsRequest
from app.services.embeddings_service import embed_problems
from app.services.extraction_service import extract_problems_for_evidence

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ─────────────────────────────────────────────────────

async def _create_job(session: AsyncSession, job_id: UUID, job_type: str) -> Job:
    """Insert a new job row into the database."""
    job = Job(id=job_id, job_type=job_type, status="pending")
    session.add(job)
    await session.commit()
    return job


async def _update_job(job_id: UUID, **updates: object) -> None:
    """Update a job row. Uses its own session for background-task safety."""
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Job).where(Job.id == job_id).values(**updates)
        )
        await session.commit()


@router.post(
    "/jobs/extract_problems",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def extract_problems_endpoint(
    payload: ExtractProblemsRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    job_id = uuid4()
    await _create_job(session, job_id, "extract_problems")
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
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    job_id = uuid4()
    await _create_job(session, job_id, "embed_problems")
    background_tasks.add_task(_run_embed_job, job_id, payload.limit)
    return JobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> JobStatusResponse:
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        job_type=job.job_type,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        result_count=job.result_count,
    )


async def _run_extract_job(
    job_id: UUID, evidence_id: UUID, max_chunks: int | None
) -> None:
    await _update_job(job_id, status="running", started_at=datetime.now(timezone.utc))
    try:
        async with AsyncSessionLocal() as session:
            mentions = await extract_problems_for_evidence(
                session, evidence_id, max_chunks, job_id=job_id
            )
        await _update_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc),
            result_count=len(mentions),
        )
    # O5 fix: removed duplicate ValueError handler — Exception already covers it
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


# ── Cost tracking endpoints ─────────────────────────────────────

@router.get("/llm/costs")
async def llm_cost_summary(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Return aggregate LLM cost metrics from DB."""
    count_q = select(func.count(LLMCallLog.id))
    cost_q = select(
        func.coalesce(func.sum(LLMCallLog.cost_usd), 0),
        func.coalesce(func.sum(LLMCallLog.input_tokens), 0),
        func.coalesce(func.sum(LLMCallLog.output_tokens), 0),
    )
    total_calls = (await session.execute(count_q)).scalar() or 0
    row = (await session.execute(cost_q)).one()
    total_cost, total_input, total_output = float(row[0]), int(row[1]), int(row[2])

    # Per-model breakdown
    model_q = select(
        LLMCallLog.model,
        func.sum(LLMCallLog.cost_usd),
    ).group_by(LLMCallLog.model)
    model_rows = (await session.execute(model_q)).all()
    by_model = {m: round(float(c), 6) for m, c in model_rows}

    return {
        "total_calls": total_calls,
        "total_cost_usd": round(total_cost, 6),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "by_model": by_model,
    }


@router.get("/llm/calls")
async def llm_call_log(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Return paginated log of LLM calls from DB."""
    total = (await session.execute(select(func.count(LLMCallLog.id)))).scalar() or 0
    query = (
        select(LLMCallLog)
        .order_by(LLMCallLog.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = (await session.execute(query)).scalars().all()
    items = [
        {
            "model": r.model,
            "operation": r.operation,
            "prompt_version": r.prompt_version,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "latency_ms": round(r.latency_ms, 1),
            "cost_usd": round(r.cost_usd, 6),
            "timestamp": r.created_at.isoformat() if r.created_at else None,
            "error": r.error,
        }
        for r in rows
    ]
    return {"items": items, "total": total, "offset": offset, "limit": limit}

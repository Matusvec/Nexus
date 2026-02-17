import asyncio
import math
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.llm.client import get_client
from app.schemas.problems import (
    ProblemMentionListResponse,
    ProblemMentionResponse,
    Severity,
    SimilarProblemResult,
    SimilarProblemsResponse,
)
from app.services.problems_service import (
    find_similar_problems,
    get_problem_mention,
    get_problem_stats,
    list_problem_mentions,
)

router = APIRouter()


@router.get("/problems/similar", response_model=SimilarProblemsResponse)
async def similar_problems_endpoint(
    text: str = Query(..., min_length=1, description="Query text to find similar problems"),
    limit: int = Query(10, ge=1, le=100),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_session),
) -> SimilarProblemsResponse:
    """Embed the query text and perform kNN search against problem embeddings."""
    client = get_client()
    query_embedding = await asyncio.to_thread(client.embed_text, text)

    results = await find_similar_problems(
        session,
        query_embedding=query_embedding,
        limit=limit,
        min_score=min_score,
    )
    return SimilarProblemsResponse(
        query_text=text,
        results=[
            SimilarProblemResult(
                problem=ProblemMentionResponse.model_validate(problem),
                score=round(score, 4),
            )
            for problem, score in results
        ],
    )


@router.get("/problems/stats")
async def problems_stats_endpoint(
    persona: str | None = None,
    severity: Severity | None = None,
    tag: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Return aggregate problem counts by severity, persona, and tag."""
    return await get_problem_stats(
        session,
        persona=persona,
        severity=severity,
        tag=tag,
    )


@router.get("/problems", response_model=ProblemMentionListResponse)
async def list_problems_endpoint(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    evidence_id: UUID | None = None,
    severity: Severity | None = None,
    persona: str | None = None,
    segment: str | None = None,
    tag: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> ProblemMentionListResponse:
    items, total = await list_problem_mentions(
        session,
        page=page,
        per_page=per_page,
        evidence_id=evidence_id,
        severity=severity,
        persona=persona,
        segment=segment,
        tag=tag,
    )
    return ProblemMentionListResponse(
        items=[ProblemMentionResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=max(1, math.ceil(total / per_page)),
    )


@router.get("/problems/{problem_id}", response_model=ProblemMentionResponse)
async def get_problem_endpoint(
    problem_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ProblemMentionResponse:
    problem = await get_problem_mention(session, problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")
    return ProblemMentionResponse.model_validate(problem)

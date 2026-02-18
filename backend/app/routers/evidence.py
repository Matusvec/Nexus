import math
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas.evidence import (
    EvidenceCreate,
    EvidenceDetailResponse,
    EvidenceListResponse,
    EvidenceResponse,
    EvidenceUpdate,
    SourceType,
)
from app.services.evidence_service import (
    create_evidence,
    delete_evidence,
    get_evidence_detail,
    list_evidence,
    update_evidence,
)

router = APIRouter()


@router.post("/evidence", response_model=EvidenceResponse, status_code=201)
async def create_evidence_endpoint(
    payload: EvidenceCreate,
    session: AsyncSession = Depends(get_session),
) -> EvidenceResponse:
    evidence, chunk_count = await create_evidence(session, payload)
    return EvidenceResponse(
        id=evidence.id,
        title=evidence.title,
        source_type=evidence.source_type,
        persona=evidence.persona,
        segment=evidence.segment,
        source_date=evidence.source_date,
        chunk_count=chunk_count,
        created_at=evidence.created_at,
    )


@router.get("/evidence", response_model=EvidenceListResponse)
async def list_evidence_endpoint(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    source_type: SourceType | None = None,
    persona: str | None = None,
    segment: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> EvidenceListResponse:
    items, total = await list_evidence(
        session,
        page=page,
        per_page=per_page,
        source_type=source_type,
        persona=persona,
        segment=segment,
    )
    return EvidenceListResponse(
        items=[EvidenceResponse(**item) for item in items],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=max(1, math.ceil(total / per_page)),
    )


@router.get("/evidence/{evidence_id}", response_model=EvidenceDetailResponse)
async def get_evidence_endpoint(
    evidence_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> EvidenceDetailResponse:
    evidence = await get_evidence_detail(session, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return EvidenceDetailResponse(
        id=evidence.id,
        title=evidence.title,
        source_type=evidence.source_type,
        persona=evidence.persona,
        segment=evidence.segment,
        source_date=evidence.source_date,
        chunk_count=len(evidence.chunks),
        created_at=evidence.created_at,
        raw_text=evidence.raw_text,
        chunks=evidence.chunks,
    )


@router.delete("/evidence/{evidence_id}", status_code=204)
async def delete_evidence_endpoint(
    evidence_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    deleted = await delete_evidence(session, evidence_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Evidence not found")


@router.put("/evidence/{evidence_id}", response_model=EvidenceResponse)
async def update_evidence_endpoint(
    evidence_id: UUID,
    payload: EvidenceUpdate,
    session: AsyncSession = Depends(get_session),
) -> EvidenceResponse:
    evidence = await update_evidence(session, evidence_id, payload)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    chunk_count = len(evidence.chunks) if evidence.chunks else 0
    return EvidenceResponse(
        id=evidence.id,
        title=evidence.title,
        source_type=evidence.source_type,
        persona=evidence.persona,
        segment=evidence.segment,
        source_date=evidence.source_date,
        chunk_count=chunk_count,
        created_at=evidence.created_at,
    )

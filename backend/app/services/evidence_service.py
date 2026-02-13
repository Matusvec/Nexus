from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.evidence import Evidence, EvidenceChunk
from app.schemas.evidence import EvidenceCreate
from app.utils.chunking import chunk_text


async def create_evidence(
    session: AsyncSession, payload: EvidenceCreate
) -> tuple[Evidence, int]:
    evidence = Evidence(
        title=payload.title,
        source_type=payload.source_type,
        persona=payload.persona,
        segment=payload.segment,
        source_date=payload.source_date,
        raw_text=payload.raw_text,
        metadata_=payload.metadata or {},
    )
    session.add(evidence)
    await session.flush()

    chunks = chunk_text(
        payload.raw_text,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    chunk_models = [
        EvidenceChunk(
            evidence_id=evidence.id,
            chunk_index=chunk["index"],
            chunk_text=chunk["text"],
            start_offset=chunk["start_offset"],
            end_offset=chunk["end_offset"],
            token_count=chunk.get("token_count"),
        )
        for chunk in chunks
    ]
    session.add_all(chunk_models)
    await session.commit()

    return evidence, len(chunk_models)


async def list_evidence(
    session: AsyncSession,
    page: int = 1,
    per_page: int = 20,
    source_type: str | None = None,
    persona: str | None = None,
    segment: str | None = None,
) -> tuple[list[dict], int]:
    """Return paginated evidence list with chunk counts."""
    # Build base query
    filters = []
    if source_type:
        filters.append(Evidence.source_type == source_type)
    if persona:
        filters.append(Evidence.persona == persona)
    if segment:
        filters.append(Evidence.segment == segment)

    # Count total
    count_q = select(func.count(Evidence.id))
    if filters:
        count_q = count_q.where(*filters)
    total = (await session.execute(count_q)).scalar() or 0

    # Fetch page with chunk counts via subquery
    chunk_count_subq = (
        select(
            EvidenceChunk.evidence_id,
            func.count(EvidenceChunk.id).label("chunk_count"),
        )
        .group_by(EvidenceChunk.evidence_id)
        .subquery()
    )

    query = (
        select(Evidence, chunk_count_subq.c.chunk_count)
        .outerjoin(chunk_count_subq, Evidence.id == chunk_count_subq.c.evidence_id)
        .order_by(Evidence.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    if filters:
        query = query.where(*filters)

    rows = (await session.execute(query)).all()

    items = []
    for evidence, chunk_count in rows:
        items.append({
            "id": evidence.id,
            "title": evidence.title,
            "source_type": evidence.source_type,
            "persona": evidence.persona,
            "segment": evidence.segment,
            "source_date": evidence.source_date,
            "chunk_count": chunk_count or 0,
            "created_at": evidence.created_at,
        })

    return items, total


async def get_evidence_detail(
    session: AsyncSession, evidence_id: UUID
) -> Evidence | None:
    """Return single evidence with its chunks eagerly loaded."""
    query = (
        select(Evidence)
        .options(selectinload(Evidence.chunks))
        .where(Evidence.id == evidence_id)
    )
    result = await session.execute(query)
    return result.scalar_one_or_none()


async def delete_evidence(
    session: AsyncSession, evidence_id: UUID
) -> bool:
    """Delete evidence by ID. Returns True if found and deleted."""
    evidence = await session.get(Evidence, evidence_id)
    if not evidence:
        return False
    await session.delete(evidence)
    await session.commit()
    return True

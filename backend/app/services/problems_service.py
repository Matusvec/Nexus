from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.embeddings import ProblemEmbedding
from app.models.problems import ProblemMention


async def list_problem_mentions(
    session: AsyncSession,
    page: int = 1,
    per_page: int = 20,
    evidence_id: UUID | None = None,
    severity: str | None = None,
    persona: str | None = None,
    segment: str | None = None,
    tag: str | None = None,
) -> tuple[list[ProblemMention], int]:
    filters = []
    if evidence_id:
        filters.append(ProblemMention.evidence_id == evidence_id)
    if severity:
        filters.append(ProblemMention.severity == severity)
    if persona:
        filters.append(ProblemMention.persona == persona)
    if segment:
        filters.append(ProblemMention.segment == segment)
    if tag:
        filters.append(ProblemMention.tags.contains([tag]))

    count_q = select(func.count(ProblemMention.id))
    if filters:
        count_q = count_q.where(*filters)
    total = (await session.execute(count_q)).scalar() or 0

    query = (
        select(ProblemMention)
        .order_by(ProblemMention.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    if filters:
        query = query.where(*filters)

    items = (await session.execute(query)).scalars().all()
    return list(items), total


async def get_problem_mention(
    session: AsyncSession, problem_id: UUID
) -> ProblemMention | None:
    return await session.get(ProblemMention, problem_id)


async def find_similar_problems(
    session: AsyncSession,
    query_embedding: list[float],
    limit: int = 10,
    min_score: float = 0.0,
) -> list[tuple[ProblemMention, float]]:
    """Find problems similar to the query embedding using cosine distance.

    Returns list of (ProblemMention, similarity_score) tuples sorted by
    descending similarity. Score is 1 - cosine_distance (higher = more similar).
    """
    # pgvector cosine distance operator: <=>
    # cosine_distance ranges from 0 (identical) to 2 (opposite)
    # We convert to similarity: 1 - distance
    distance = ProblemEmbedding.embedding.cosine_distance(query_embedding)

    query = (
        select(ProblemMention, (1 - distance).label("similarity"))
        .join(ProblemEmbedding, ProblemMention.id == ProblemEmbedding.problem_id)
        .order_by(distance.asc())
        .limit(limit)
    )

    rows = (await session.execute(query)).all()

    results = []
    for problem, similarity in rows:
        if similarity >= min_score:
            results.append((problem, float(similarity)))
    return results


async def get_problem_stats(
    session: AsyncSession,
    persona: str | None = None,
    severity: str | None = None,
    tag: str | None = None,
) -> dict:
    """Return aggregate counts grouped by severity, persona, and tag."""
    base_filters = []
    if persona:
        base_filters.append(ProblemMention.persona == persona)
    if severity:
        base_filters.append(ProblemMention.severity == severity)
    if tag:
        base_filters.append(ProblemMention.tags.contains([tag]))

    # Total count
    total_q = select(func.count(ProblemMention.id))
    if base_filters:
        total_q = total_q.where(*base_filters)
    total = (await session.execute(total_q)).scalar() or 0

    # By severity
    sev_q = (
        select(ProblemMention.severity, func.count(ProblemMention.id))
        .group_by(ProblemMention.severity)
    )
    if base_filters:
        sev_q = sev_q.where(*base_filters)
    sev_rows = (await session.execute(sev_q)).all()
    by_severity = {sev: cnt for sev, cnt in sev_rows}

    # By persona
    persona_q = (
        select(ProblemMention.persona, func.count(ProblemMention.id))
        .where(ProblemMention.persona.isnot(None))
        .group_by(ProblemMention.persona)
    )
    if base_filters:
        persona_q = persona_q.where(*base_filters)
    persona_rows = (await session.execute(persona_q)).all()
    by_persona = {p: cnt for p, cnt in persona_rows}

    # By tag (unnest the array)
    tag_q = select(
        func.unnest(ProblemMention.tags).label("tag"),
        func.count(ProblemMention.id),
    ).group_by("tag")
    if base_filters:
        tag_q = tag_q.where(*base_filters)
    tag_rows = (await session.execute(tag_q)).all()
    by_tag = {t: cnt for t, cnt in tag_rows}

    return {
        "total": total,
        "by_severity": by_severity,
        "by_persona": by_persona,
        "by_tag": by_tag,
    }

import asyncio
import logging
from typing import Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.llm.client import get_client
from app.models.embeddings import ProblemEmbedding
from app.models.problems import ProblemMention

logger = logging.getLogger(__name__)
MAX_CONCURRENCY = 4
MAX_RETRIES = 3

def _embedding_text(problem: ProblemMention) -> str:
    return f"{problem.problem_statement}\n\nQuote: {problem.quote_text}"


async def embed_problems(
    session: AsyncSession,
    limit: int | None = None,
) -> list[UUID]:
    query = (
        select(ProblemMention)
        .outerjoin(ProblemEmbedding, ProblemMention.id == ProblemEmbedding.problem_id)
        .where(ProblemEmbedding.id.is_(None))
        .order_by(ProblemMention.created_at.desc())
    )
    if limit:
        query = query.limit(limit)
    problems: Iterable[ProblemMention] = (await session.execute(query)).scalars().all()

    if not problems:
        return []

    client = get_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process_problem(problem: ProblemMention) -> ProblemEmbedding | None:
        async with semaphore:
            text = _embedding_text(problem)
            try:
                vector = await _call_with_retry(client.embed_text, text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding failed for problem %s: %s", problem.id, exc)
                return None
            return ProblemEmbedding(
                problem_id=problem.id,
                embedding=vector,
                model_version=settings.gemini_embedding_model,
            )

    results = await asyncio.gather(
        *(process_problem(problem) for problem in problems),
        return_exceptions=False,
    )
    embeddings = [item for item in results if item is not None]
    if not embeddings:
        return []

    session.add_all(embeddings)
    await session.commit()
    return [embedding.problem_id for embedding in embeddings]


async def _call_with_retry(func, text: str) -> list[float]:
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(func, text)
        except Exception as exc:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                raise
            logger.info("Embedding call failed (attempt %s): %s", attempt, exc)
            await asyncio.sleep(delay)
            delay *= 2

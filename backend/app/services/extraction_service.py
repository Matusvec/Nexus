import asyncio
import logging
from uuid import UUID

from sqlalchemy import delete, exists, select
from sqlalchemy.ext.asyncio import AsyncSession
from thefuzz import fuzz

from app.llm.client import get_client
from app.models.evidence import Evidence, EvidenceChunk
from app.models.problems import ProblemMention
from app.schemas.problems import LLMProblemsResponse, ProblemMentionCreate
from app.utils.retry import call_with_retry

PROMPT_VERSION = "extract_problems_v1"
logger = logging.getLogger(__name__)
MAX_CONCURRENCY = 4
FUZZY_MATCH_THRESHOLD = 70  # minimum partial_ratio score to accept a fuzzy match


def _build_prompt(chunk_text: str) -> str:
    return (
        "You are extracting customer problems from a transcript chunk.\n"
        "Return valid JSON only, with this schema:\n"
        "{\n"
        '  "problems": [\n'
        "    {\n"
        '      "problem_statement": "string",\n'
        '      "severity": "critical|high|medium|low",\n'
        '      "quote_text": "direct quote from the chunk",\n'
        '      "persona": "optional",\n'
        '      "segment": "optional",\n'
        '      "tags": ["tag1", "tag2"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "If no problems are present, return {\"problems\": []}.\n\n"
        f"Chunk:\n{chunk_text}\n"
    )


def _find_quote_offsets(
    chunk_text: str, quote_text: str
) -> tuple[int | None, int | None, bool]:
    """Locate quote_text within chunk_text.

    Returns (start, end, verified).
    - Exact match: returns exact offsets, verified=True
    - Fuzzy match above threshold: returns best-fit offsets, verified=True
    - No match: returns (None, None, False)
    """
    if not quote_text:
        return None, None, False

    # 1. Try exact match first
    start = chunk_text.find(quote_text)
    if start != -1:
        return start, start + len(quote_text), True

    # 2. Fuzzy sliding-window search
    quote_len = len(quote_text)
    best_score = 0
    best_start = -1

    # Slide a window roughly the size of the quote across the chunk
    # Use a step size for performance on large chunks
    step = max(1, quote_len // 10)
    for window_start in range(0, max(1, len(chunk_text) - quote_len // 2), step):
        window_end = min(len(chunk_text), window_start + quote_len + quote_len // 4)
        window = chunk_text[window_start:window_end]
        score = fuzz.partial_ratio(quote_text.lower(), window.lower())
        if score > best_score:
            best_score = score
            best_start = window_start

    if best_score >= FUZZY_MATCH_THRESHOLD and best_start >= 0:
        best_end = min(len(chunk_text), best_start + quote_len)
        logger.info(
            "Fuzzy quote match (score=%d): expected=%r, found window at [%d:%d]",
            best_score, quote_text[:60], best_start, best_end,
        )
        return best_start, best_end, True

    # 3. No match
    logger.warning(
        "Quote verification FAILED (best_score=%d): quote=%r not found in chunk (len=%d)",
        best_score, quote_text[:80], len(chunk_text),
    )
    return None, None, False


async def extract_problems_for_evidence(
    session: AsyncSession,
    evidence_id: UUID,
    max_chunks: int | None = None,
    job_id: UUID | None = None,
) -> list[ProblemMention]:
    evidence = await session.get(Evidence, evidence_id)
    if not evidence:
        raise ValueError("Evidence not found")

    await _clear_existing_mentions(session, evidence_id)

    query = (
        select(EvidenceChunk)
        .where(EvidenceChunk.evidence_id == evidence_id)
        .order_by(EvidenceChunk.chunk_index.asc())
    )
    if max_chunks:
        query = query.limit(max_chunks)
    chunks: list[EvidenceChunk] = (await session.execute(query)).scalars().all()

    client = get_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process_chunk(chunk: EvidenceChunk) -> list[ProblemMention]:
        async with semaphore:
            prompt = _build_prompt(chunk.chunk_text)
            try:
                raw = await call_with_retry(
                    client.generate_json, prompt, label="Extraction LLM call"
                )
                parsed = LLMProblemsResponse.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Extraction failed for chunk %s: %s", chunk.id, exc
                )
                return []
            return [
                m
                for m in (
                    _build_problem_mention(evidence, chunk, problem, job_id=job_id)
                    for problem in parsed.problems
                )
                if m is not None
            ]

    results = await asyncio.gather(
        *(process_chunk(chunk) for chunk in chunks),
        return_exceptions=False,
    )
    created_mentions = [item for sublist in results for item in sublist]

    if created_mentions:
        session.add_all(created_mentions)
        await session.commit()
    else:
        logger.info("No problems extracted for evidence %s", evidence_id)
    return created_mentions


async def _clear_existing_mentions(session: AsyncSession, evidence_id: UUID) -> None:
    # C2 fix: warn if clearing mentions will cascade to embeddings/clusters
    from app.models.embeddings import ProblemEmbedding
    from app.models.clusters import ClusterMembership

    has_embeddings = (
        await session.execute(
            select(
                exists(
                    select(ProblemEmbedding.id)
                    .join(ProblemMention, ProblemEmbedding.problem_id == ProblemMention.id)
                    .where(ProblemMention.evidence_id == evidence_id)
                )
            )
        )
    ).scalar()

    if has_embeddings:
        has_clusters = (
            await session.execute(
                select(
                    exists(
                        select(ClusterMembership.id)
                        .join(ProblemMention, ClusterMembership.problem_id == ProblemMention.id)
                        .where(ProblemMention.evidence_id == evidence_id)
                    )
                )
            )
        ).scalar()
        if has_clusters:
            logger.warning(
                "Re-extraction for evidence %s will CASCADE-delete embeddings AND "
                "cluster memberships. Clusters should be re-run after this.",
                evidence_id,
            )
        else:
            logger.warning(
                "Re-extraction for evidence %s will CASCADE-delete associated embeddings.",
                evidence_id,
            )

    await session.execute(
        delete(ProblemMention).where(ProblemMention.evidence_id == evidence_id)
    )
    # M4 fix: use flush() instead of commit() so the outer function
    # can commit everything atomically (prevents data loss on LLM failure)
    await session.flush()


def _build_problem_mention(
    evidence: Evidence,
    chunk: EvidenceChunk,
    problem: ProblemMentionCreate,
    *,
    job_id: UUID | None = None,
) -> ProblemMention | None:
    """Build a ProblemMention, verifying the quote exists in the chunk.

    Returns None if the quote cannot be verified (exact or fuzzy),
    effectively dropping hallucinated quotes.
    """
    quote_start, quote_end, verified = _find_quote_offsets(
        chunk.chunk_text, problem.quote_text
    )
    if not verified:
        logger.warning(
            "Dropping unverified problem mention (evidence=%s, chunk=%s): %s",
            evidence.id, chunk.id, problem.problem_statement[:80],
        )
        return None

    return ProblemMention(
        evidence_id=evidence.id,
        chunk_id=chunk.id,
        problem_statement=problem.problem_statement,
        persona=problem.persona or evidence.persona,
        segment=problem.segment or evidence.segment,
        severity=problem.severity,
        quote_text=problem.quote_text,
        quote_start=quote_start,
        quote_end=quote_end,
        tags=problem.tags or [],
        prompt_version=PROMPT_VERSION,
        extraction_job_id=job_id,
    )

import asyncio
import logging
from typing import Iterable
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from thefuzz import fuzz

from app.llm.client import get_client
from app.models.evidence import Evidence, EvidenceChunk
from app.models.problems import ProblemMention
from app.schemas.problems import LLMProblemsResponse, ProblemMentionCreate

PROMPT_VERSION = "extract_problems_v1"
logger = logging.getLogger(__name__)
MAX_CONCURRENCY = 4
MAX_RETRIES = 3
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
    chunks: Iterable[EvidenceChunk] = (await session.execute(query)).scalars().all()

    client = get_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process_chunk(chunk: EvidenceChunk) -> list[ProblemMention]:
        async with semaphore:
            prompt = _build_prompt(chunk.chunk_text)
            try:
                raw = await _call_with_retry(client.generate_json, prompt)
                parsed = LLMProblemsResponse.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Extraction failed for chunk %s: %s", chunk.id, exc
                )
                return []
            return [
                m
                for m in (
                    _build_problem_mention(evidence, chunk, problem)
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
        await session.flush()
        await session.commit()
    else:
        logger.info("No problems extracted for evidence %s", evidence_id)
    return created_mentions


async def _call_with_retry(func, prompt: str) -> dict:
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(func, prompt)
        except Exception as exc:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                raise
            logger.info("LLM call failed (attempt %s): %s", attempt, exc)
            await asyncio.sleep(delay)
            delay *= 2


async def _clear_existing_mentions(session: AsyncSession, evidence_id: UUID) -> None:
    await session.execute(
        delete(ProblemMention).where(ProblemMention.evidence_id == evidence_id)
    )
    await session.commit()


def _build_problem_mention(
    evidence: Evidence,
    chunk: EvidenceChunk,
    problem: ProblemMentionCreate,
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
    )

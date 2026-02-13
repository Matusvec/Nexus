import asyncio
from typing import Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import GeminiClient
from app.models.evidence import Evidence, EvidenceChunk
from app.models.problems import ProblemMention
from app.schemas.problems import LLMProblemsResponse, ProblemMentionCreate

PROMPT_VERSION = "extract_problems_v1"


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


def _find_quote_offsets(chunk_text: str, quote_text: str) -> tuple[int | None, int | None]:
    if not quote_text:
        return None, None
    start = chunk_text.find(quote_text)
    if start == -1:
        return None, None
    return start, start + len(quote_text)


async def extract_problems_for_evidence(
    session: AsyncSession,
    evidence_id: UUID,
    max_chunks: int | None = None,
) -> list[ProblemMention]:
    evidence = await session.get(Evidence, evidence_id)
    if not evidence:
        raise ValueError("Evidence not found")

    query = (
        select(EvidenceChunk)
        .where(EvidenceChunk.evidence_id == evidence_id)
        .order_by(EvidenceChunk.chunk_index.asc())
    )
    if max_chunks:
        query = query.limit(max_chunks)
    chunks: Iterable[EvidenceChunk] = (await session.execute(query)).scalars().all()

    client = GeminiClient()
    created_mentions: list[ProblemMention] = []

    for chunk in chunks:
        prompt = _build_prompt(chunk.chunk_text)
        raw = await asyncio.to_thread(client.generate_json, prompt)
        parsed = LLMProblemsResponse.model_validate(raw)
        for problem in parsed.problems:
            created_mentions.append(
                _build_problem_mention(evidence, chunk, problem)
            )

    if not created_mentions:
        return []

    session.add_all(created_mentions)
    await session.flush()
    await session.commit()
    return created_mentions


def _build_problem_mention(
    evidence: Evidence,
    chunk: EvidenceChunk,
    problem: ProblemMentionCreate,
) -> ProblemMention:
    quote_start, quote_end = _find_quote_offsets(chunk.chunk_text, problem.quote_text)
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

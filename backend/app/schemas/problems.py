from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

Severity = Literal["critical", "high", "medium", "low"]


class ExtractProblemsRequest(BaseModel):
    evidence_id: UUID
    max_chunks: int | None = None


class ExtractProblemsResponse(BaseModel):
    evidence_id: UUID
    extracted_count: int
    problems: list["ProblemMentionResponse"] = []


class ProblemMentionResponse(BaseModel):
    id: UUID
    evidence_id: UUID
    chunk_id: UUID
    problem_statement: str
    severity: Severity
    quote_text: str
    quote_start: int | None = None
    quote_end: int | None = None
    persona: str | None = None
    segment: str | None = None
    tags: list[str] = []
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ProblemMentionCreate(BaseModel):
    problem_statement: str
    severity: Severity
    quote_text: str
    persona: str | None = None
    segment: str | None = None
    tags: list[str] = []


class LLMProblemsResponse(BaseModel):
    problems: list[ProblemMentionCreate] = []


class ProblemMentionListResponse(BaseModel):
    items: list[ProblemMentionResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


class SimilarProblemResult(BaseModel):
    """A problem mention returned from similarity search, with distance score."""
    problem: ProblemMentionResponse
    score: float  # cosine similarity (0–1, higher = more similar)


class SimilarProblemsResponse(BaseModel):
    query_text: str
    results: list[SimilarProblemResult]

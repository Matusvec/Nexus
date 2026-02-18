from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["interview", "support_ticket", "sales_note", "survey", "other"]


class EvidenceCreate(BaseModel):
    title: str
    source_type: SourceType
    persona: str | None = None
    segment: str | None = None
    source_date: date | None = None
    raw_text: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class EvidenceChunkResponse(BaseModel):
    id: UUID
    chunk_index: int
    chunk_text: str
    start_offset: int
    end_offset: int
    token_count: int | None = None

    model_config = ConfigDict(from_attributes=True)


class EvidenceResponse(BaseModel):
    id: UUID
    title: str
    source_type: str
    persona: str | None = None
    segment: str | None = None
    source_date: date | None = None
    chunk_count: int
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class EvidenceDetailResponse(EvidenceResponse):
    """Full evidence detail including raw text and chunks."""
    raw_text: str
    chunks: list[EvidenceChunkResponse] = []


class EvidenceUpdate(BaseModel):
    title: str | None = None
    source_type: SourceType | None = None
    persona: str | None = None
    segment: str | None = None
    source_date: date | None = None
    metadata: dict | None = None
    # NOTE: raw_text is NOT updatable — changing text would invalidate all
    # chunks, problem mentions, embeddings, and clusters downstream.
    # If text changes, delete + re-upload.


class EvidenceListResponse(BaseModel):
    items: list[EvidenceResponse]
    total: int
    page: int
    per_page: int
    total_pages: int

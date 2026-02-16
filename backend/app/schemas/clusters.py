from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


# ── Cluster schemas ──────────────────────────────────────────────

class ClusterBase(BaseModel):
    label: str
    summary: str | None = None
    threshold: float = 0.75
    tags: list[str] = []


class ClusterCreate(ClusterBase):
    pass


class ClusterResponse(ClusterBase):
    id: UUID
    mention_count: int
    created_at: datetime | None = None
    updated_at: datetime | None = None
    model_config = ConfigDict(from_attributes=True)


class ClusterDetailResponse(ClusterResponse):
    members: list["ClusterMemberResponse"] = []
    proposals: list["ProposalResponse"] = []


class ClusterMemberResponse(BaseModel):
    id: UUID
    problem_id: UUID
    similarity: float
    model_config = ConfigDict(from_attributes=True)


# ── Proposal schemas ────────────────────────────────────────────

class ProposalBase(BaseModel):
    title: str
    description: str
    priority_score: float | None = None
    impact: str | None = None
    effort: str | None = None


class ProposalCreate(ProposalBase):
    cluster_id: UUID


class ProposalResponse(ProposalBase):
    id: UUID
    cluster_id: UUID
    version: int
    created_at: datetime | None = None
    updated_at: datetime | None = None
    model_config = ConfigDict(from_attributes=True)


class ProposalDetailResponse(ProposalResponse):
    citations: list["CitationResponse"] = []


class CitationResponse(BaseModel):
    id: UUID
    problem_id: UUID
    relevance_note: str | None = None
    model_config = ConfigDict(from_attributes=True)


# ── Roadmap schema ──────────────────────────────────────────────

class RoadmapItem(BaseModel):
    proposal: ProposalResponse
    cluster_label: str
    mention_count: int
    priority_score: float | None = None


class RoadmapResponse(BaseModel):
    items: list[RoadmapItem]
    total: int

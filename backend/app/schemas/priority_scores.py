"""Pydantic schemas for priority scores and roadmap ranking."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class PriorityScoreResponse(BaseModel):
    id: UUID
    proposal_id: UUID
    frequency_score: float
    severity_score: float
    strategic_weight: float
    effort_estimate: float
    final_score: float
    score_breakdown: dict
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class StrategicWeightUpdate(BaseModel):
    """Payload for adjusting a proposal's strategic weight."""
    strategic_weight: float

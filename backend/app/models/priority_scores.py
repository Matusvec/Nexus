"""Priority score model — strategy Section G.

Stores explainable scoring breakdown for each feature proposal,
enabling transparent roadmap ranking.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from app.database import Base


class PriorityScore(Base):
    """Explainable priority score attached to a feature proposal."""

    __tablename__ = "priority_scores"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    proposal_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("feature_proposals.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    frequency_score: Mapped[float] = mapped_column(Float, nullable=False)
    severity_score: Mapped[float] = mapped_column(Float, nullable=False)
    strategic_weight: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("1.0")
    )
    effort_estimate: Mapped[float] = mapped_column(Float, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    score_breakdown: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    proposal = relationship("FeatureProposal", backref="priority_score_rel")

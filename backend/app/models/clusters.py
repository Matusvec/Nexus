from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Float, Integer, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from app.database import Base


class ProblemCluster(Base):
    """A group of semantically similar problem mentions."""

    __tablename__ = "problem_clusters"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    centroid: Mapped[list[float] | None] = mapped_column(Vector(768))
    threshold: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0.75"))
    mention_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text), server_default=text("ARRAY[]::text[]")
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb")  # O2: reserved for extensibility
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    members = relationship(
        "ClusterMembership", back_populates="cluster", cascade="all, delete-orphan"
    )
    proposals = relationship(
        "FeatureProposal", back_populates="cluster", cascade="all, delete-orphan"
    )


class ClusterMembership(Base):
    """Junction between problem_mentions and clusters."""

    __tablename__ = "cluster_memberships"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    cluster_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("problem_clusters.id", ondelete="CASCADE"),
        nullable=False,
    )
    problem_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("problem_mentions.id", ondelete="CASCADE"),
        nullable=False,
    )
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    cluster = relationship("ProblemCluster", back_populates="members")
    problem = relationship("ProblemMention")


class FeatureProposal(Base):
    """A feature proposal derived from a problem cluster."""

    __tablename__ = "feature_proposals"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    cluster_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("problem_clusters.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    priority_score: Mapped[float | None] = mapped_column(Float)
    impact: Mapped[str | None] = mapped_column(Text)  # high/medium/low
    effort: Mapped[str | None] = mapped_column(Text)  # high/medium/low
    version: Mapped[int] = mapped_column(Integer, server_default=text("1"))  # O1: placeholder for future proposal versioning
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'draft'"))
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb")  # O2: reserved for extensibility
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    cluster = relationship("ProblemCluster", back_populates="proposals")
    citations = relationship(
        "ProposalCitation", back_populates="proposal", cascade="all, delete-orphan"
    )


class ProposalVersion(Base):
    """Snapshot of a proposal at a specific version for auditability."""

    __tablename__ = "proposal_versions"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    proposal_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("feature_proposals.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    change_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    proposal = relationship("FeatureProposal")


class ProposalCitation(Base):
    """Links a proposal back to the problem mentions that justify it."""

    __tablename__ = "proposal_citations"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    proposal_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("feature_proposals.id", ondelete="CASCADE"),
        nullable=False,
    )
    problem_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("problem_mentions.id", ondelete="CASCADE"),
        nullable=False,
    )
    relevance_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    proposal = relationship("FeatureProposal", back_populates="citations")
    problem = relationship("ProblemMention")

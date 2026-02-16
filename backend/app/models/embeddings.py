from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from app.database import Base


class ProblemEmbedding(Base):
    __tablename__ = "problem_embeddings"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    problem_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("problem_mentions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    problem = relationship("ProblemMention", back_populates="embedding")

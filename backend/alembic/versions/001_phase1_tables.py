"""Phase 1 tables: evidence, evidence_chunks, problem_mentions, problem_embeddings

Revision ID: 001_phase1_tables
Revises: None
Create Date: 2026-01-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_phase1_tables"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- evidence ---
    op.create_table(
        "evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("persona", sa.Text(), nullable=True),
        sa.Column("segment", sa.Text(), nullable=True),
        sa.Column("source_date", sa.Date(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- evidence_chunks ---
    op.create_table(
        "evidence_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("evidence_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_evidence_chunks_evidence_id", "evidence_chunks", ["evidence_id"])

    # --- problem_mentions ---
    op.create_table(
        "problem_mentions",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("evidence_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("evidence_chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("problem_statement", sa.Text(), nullable=False),
        sa.Column("persona", sa.Text(), nullable=True),
        sa.Column("segment", sa.Text(), nullable=True),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("quote_text", sa.Text(), nullable=False),
        sa.Column("quote_start", sa.Integer(), nullable=True),
        sa.Column("quote_end", sa.Integer(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), server_default=sa.text("ARRAY[]::text[]"), nullable=True),
        sa.Column("extraction_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("prompt_version", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # Indexes for common filter columns
    op.create_index("ix_problem_mentions_evidence_id", "problem_mentions", ["evidence_id"])
    op.create_index("ix_problem_mentions_severity", "problem_mentions", ["severity"])
    op.create_index("ix_problem_mentions_persona", "problem_mentions", ["persona"])
    # GIN index for tags array
    op.execute(
        "CREATE INDEX ix_problem_mentions_tags ON problem_mentions USING GIN (tags)"
    )

    # --- problem_embeddings ---
    # Create table with basic columns first, then add vector column via raw SQL
    op.create_table(
        "problem_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("problem_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("problem_mentions.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # Add vector column via raw SQL (768 dimensions to match text-embedding-004)
    op.execute("ALTER TABLE problem_embeddings ADD COLUMN embedding vector(768) NOT NULL")
    # HNSW index for cosine similarity search (better than ivfflat for smaller datasets)
    op.execute(
        "CREATE INDEX ix_problem_embeddings_embedding ON problem_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("problem_embeddings")
    op.drop_table("problem_mentions")
    op.drop_table("evidence_chunks")
    op.drop_table("evidence")
    op.execute("DROP EXTENSION IF EXISTS vector")

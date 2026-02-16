"""Phase 2 tables: clusters, memberships, proposals, citations

Revision ID: 002_phase2_clusters
Revises: 001_phase1_tables
Create Date: 2026-01-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "002_phase2_clusters"
down_revision: Union[str, None] = "001_phase1_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- problem_clusters ---
    op.create_table(
        "problem_clusters",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("threshold", sa.Float(), server_default=sa.text("0.75"), nullable=False),
        sa.Column("mention_count", sa.Integer(), server_default=sa.text("0"), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), server_default=sa.text("ARRAY[]::text[]"), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # Add centroid vector column
    op.execute("ALTER TABLE problem_clusters ADD COLUMN centroid vector(768)")

    # --- cluster_memberships ---
    op.create_table(
        "cluster_memberships",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("cluster_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("problem_clusters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("problem_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("problem_mentions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_cluster_memberships_cluster_id", "cluster_memberships", ["cluster_id"])
    op.create_index("ix_cluster_memberships_problem_id", "cluster_memberships", ["problem_id"])

    # --- feature_proposals ---
    op.create_table(
        "feature_proposals",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("cluster_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("problem_clusters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("priority_score", sa.Float(), nullable=True),
        sa.Column("impact", sa.Text(), nullable=True),
        sa.Column("effort", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_feature_proposals_cluster_id", "feature_proposals", ["cluster_id"])

    # --- proposal_citations ---
    op.create_table(
        "proposal_citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("proposal_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("feature_proposals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("problem_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("problem_mentions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relevance_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_proposal_citations_proposal_id", "proposal_citations", ["proposal_id"])


def downgrade() -> None:
    op.drop_table("proposal_citations")
    op.drop_table("feature_proposals")
    op.drop_table("cluster_memberships")
    op.drop_table("problem_clusters")

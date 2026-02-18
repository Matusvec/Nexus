"""Add status column to feature_proposals.

Supports proposal lifecycle: draft → approved / rejected / archived.

Revision ID: 005_add_proposal_status
Revises: 004_phase4_tasks_priorities
Create Date: 2026-02-18
"""

from alembic import op
import sqlalchemy as sa

revision = "005_add_proposal_status"
down_revision = "004_phase4_tasks_priorities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "feature_proposals",
        sa.Column("status", sa.Text(), server_default="draft", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("feature_proposals", "status")

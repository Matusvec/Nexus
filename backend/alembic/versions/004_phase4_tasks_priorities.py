"""Phase 4: add tasks, priority_scores, proposal_versions tables.

Implements remaining tables from the Backend Strategy Document:
- tasks: hierarchical task trees for feature proposals (Section F)
- priority_scores: explainable scoring for roadmap ranking (Section G)
- proposal_versions: proposal snapshots for auditability (Section E)

Revision ID: 004_phase4_tasks_priorities
Revises: 003_phase3_jobs_llm
Create Date: 2026-02-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "004_phase4_tasks_priorities"
down_revision = "003_phase3_jobs_llm"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── tasks table (Strategy Section F) ─────────────────────
    op.create_table(
        "tasks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("proposal_id", UUID(as_uuid=True), sa.ForeignKey("feature_proposals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_task_id", UUID(as_uuid=True), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("acceptance_criteria", JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("estimated_effort", sa.Text(), nullable=True),
        sa.Column("dependencies", ARRAY(UUID(as_uuid=True)), server_default=sa.text("ARRAY[]::uuid[]")),
        sa.Column("sort_order", sa.Integer(), server_default=sa.text("0")),
        sa.Column("prompt_version", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_tasks_proposal_id", "tasks", ["proposal_id"])
    op.create_index("ix_tasks_parent_task_id", "tasks", ["parent_task_id"])
    op.create_index("ix_tasks_category", "tasks", ["category"])

    # ── priority_scores table (Strategy Section G) ───────────
    op.create_table(
        "priority_scores",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("proposal_id", UUID(as_uuid=True), sa.ForeignKey("feature_proposals.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("frequency_score", sa.Float(), nullable=False),
        sa.Column("severity_score", sa.Float(), nullable=False),
        sa.Column("strategic_weight", sa.Float(), server_default=sa.text("1.0"), nullable=False),
        sa.Column("effort_estimate", sa.Float(), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("score_breakdown", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_priority_scores_final_score", "priority_scores", [sa.text("final_score DESC")])

    # ── proposal_versions table (Strategy Section E) ─────────
    op.create_table(
        "proposal_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("proposal_id", UUID(as_uuid=True), sa.ForeignKey("feature_proposals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("snapshot", JSONB(), nullable=False),
        sa.Column("change_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_proposal_versions_proposal_id", "proposal_versions", ["proposal_id"])

    # ── Add updated_at trigger for priority_scores ───────────
    op.execute("""
        CREATE TRIGGER set_updated_at
        BEFORE UPDATE ON priority_scores
        FOR EACH ROW
        EXECUTE FUNCTION trigger_set_updated_at();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS set_updated_at ON priority_scores;")
    op.drop_table("proposal_versions")
    op.drop_table("priority_scores")
    op.drop_table("tasks")

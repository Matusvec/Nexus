"""Phase 3: add jobs and llm_call_log tables.

Revision ID: 003_phase3_jobs_llm
Revises: 002_phase2_clusters
Create Date: 2025-01-16 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "003_phase3_jobs_llm"
down_revision = "002_phase2_clusters"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Jobs table (A1 fix) ──────────────────────────────────
    op.create_table(
        "jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result_count", sa.Integer(), nullable=True),
        sa.Column("meta", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])

    # ── LLM call log table (A2 fix) ─────────────────────────
    op.create_table(
        "llm_call_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), server_default=sa.text("0")),
        sa.Column("output_tokens", sa.Integer(), server_default=sa.text("0")),
        sa.Column("latency_ms", sa.Float(), server_default=sa.text("0")),
        sa.Column("cost_usd", sa.Float(), server_default=sa.text("0")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_llm_call_log_created_at", "llm_call_log", ["created_at"])

    # ── CHECK constraints for existing tables (M9, M10 fixes) ──
    op.create_check_constraint(
        "ck_evidence_source_type",
        "evidence",
        "source_type IN ('interview', 'support_ticket', 'sales_note', 'survey', 'other')",
    )
    op.create_check_constraint(
        "ck_problem_mentions_severity",
        "problem_mentions",
        "severity IN ('critical', 'high', 'medium', 'low')",
    )

    # ── m8 fix: updated_at trigger at DB level ─────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION trigger_set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    for table in ("evidence", "problem_clusters", "feature_proposals"):
        op.execute(f"""
            CREATE TRIGGER set_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION trigger_set_updated_at();
        """)


def downgrade() -> None:
    for table in ("feature_proposals", "problem_clusters", "evidence"):
        op.execute(f"DROP TRIGGER IF EXISTS set_updated_at ON {table};")
    op.execute("DROP FUNCTION IF EXISTS trigger_set_updated_at();")
    op.drop_constraint("ck_problem_mentions_severity", "problem_mentions", type_="check")
    op.drop_constraint("ck_evidence_source_type", "evidence", type_="check")
    op.drop_table("llm_call_log")
    op.drop_table("jobs")

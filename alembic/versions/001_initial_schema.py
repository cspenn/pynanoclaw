# start alembic/versions/001_initial_schema.py
"""Initial database schema for NanoClaw.

Creates all 6 core tables that replace the inline createSchema() from db.ts.

Revision ID: 001
Revises: (none)
Create Date: 2026-02-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create all initial tables and indexes."""
    op.create_table(
        "chats",
        sa.Column("jid", sa.Text(), nullable=False, primary_key=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("last_message_time", sa.Text(), nullable=True),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("chat_jid", sa.Text(), sa.ForeignKey("chats.jid"), nullable=False),
        sa.Column("sender", sa.Text(), nullable=True),
        sa.Column("sender_name", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.Text(), nullable=True),
        sa.Column("is_from_me", sa.Integer(), nullable=True),
        sa.Column("is_bot_message", sa.Integer(), server_default="0", nullable=True),
        sa.PrimaryKeyConstraint("id", "chat_jid"),
    )
    op.create_index("idx_timestamp", "messages", ["timestamp"])

    op.create_table(
        "scheduled_tasks",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("group_folder", sa.Text(), nullable=False),
        sa.Column("chat_jid", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("schedule_type", sa.Text(), nullable=False),
        sa.Column("schedule_value", sa.Text(), nullable=False),
        sa.Column("context_mode", sa.Text(), server_default="isolated", nullable=True),
        sa.Column("next_run", sa.Text(), nullable=True),
        sa.Column("last_run", sa.Text(), nullable=True),
        sa.Column("last_result", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default="active", nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index("idx_next_run", "scheduled_tasks", ["next_run"])
    op.create_index("idx_status", "scheduled_tasks", ["status"])

    op.create_table(
        "task_run_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, primary_key=True),
        sa.Column("task_id", sa.Text(), sa.ForeignKey("scheduled_tasks.id"), nullable=False),
        sa.Column("run_at", sa.Text(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("idx_task_run_logs", "task_run_logs", ["task_id", "run_at"])

    op.create_table(
        "router_state",
        sa.Column("key", sa.Text(), nullable=False, primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
    )

    op.create_table(
        "sessions",
        sa.Column("group_folder", sa.Text(), nullable=False, primary_key=True),
        sa.Column("session_id", sa.Text(), nullable=False),
    )

    op.create_table(
        "registered_groups",
        sa.Column("jid", sa.Text(), nullable=False, primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("folder", sa.Text(), nullable=False, unique=True),
        sa.Column("trigger_pattern", sa.Text(), nullable=False),
        sa.Column("added_at", sa.Text(), nullable=False),
        sa.Column("container_config", sa.Text(), nullable=True),
        sa.Column("requires_trigger", sa.Integer(), server_default="1", nullable=True),
    )


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.drop_table("registered_groups")
    op.drop_table("sessions")
    op.drop_table("router_state")
    op.drop_index("idx_task_run_logs", "task_run_logs")
    op.drop_table("task_run_logs")
    op.drop_index("idx_status", "scheduled_tasks")
    op.drop_index("idx_next_run", "scheduled_tasks")
    op.drop_table("scheduled_tasks")
    op.drop_index("idx_timestamp", "messages")
    op.drop_table("messages")
    op.drop_table("chats")
# end alembic/versions/001_initial_schema.py

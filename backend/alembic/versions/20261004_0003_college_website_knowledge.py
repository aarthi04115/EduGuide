"""Add shared official college website sources and chunks."""

from alembic import op
import sqlalchemy as sa


revision = "20261004_0003"
down_revision = "20261004_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "college_sources",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=False),
        sa.Column("page_title", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "content_text",
            sa.Text(),
            server_default="",
            nullable=False,
        ),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("duplicate_of_id", sa.String(length=64), nullable=True),
        sa.Column("last_fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.String(length=64), nullable=True),
        sa.CheckConstraint(
            "status IN ('indexed', 'stale', 'failed', 'duplicate')",
            name="ck_college_sources_status",
        ),
        sa.CheckConstraint(
            "source_type IN ('html', 'pdf')",
            name="ck_college_sources_source_type",
        ),
        sa.ForeignKeyConstraint(
            ["duplicate_of_id"],
            ["college_sources.id"],
            name="fk_college_sources_duplicate_of_id_college_sources",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_url", name="uq_college_sources_url"),
    )
    op.create_index(
        "ix_college_sources_status",
        "college_sources",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_college_sources_content_hash",
        "college_sources",
        ["content_hash"],
        unique=False,
    )
    op.create_table(
        "college_chunks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["college_sources.id"],
            name="fk_college_chunks_source_id_college_sources",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_id",
            "chunk_index",
            name="uq_college_chunks_source_index",
        ),
    )
    op.create_index(
        "ix_college_chunks_source",
        "college_chunks",
        ["source_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_college_chunks_source", table_name="college_chunks")
    op.drop_table("college_chunks")
    op.drop_index("ix_college_sources_content_hash", table_name="college_sources")
    op.drop_index("ix_college_sources_status", table_name="college_sources")
    op.drop_table("college_sources")

"""Associate private study materials with conversations."""

from alembic import op
import sqlalchemy as sa


revision = "20261004_0002"
down_revision = "20261004_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "sources",
            sa.JSON(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
    )
    op.create_table(
        "conversation_documents",
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_conversation_documents_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_conversation_documents_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("conversation_id", "document_id"),
    )
    op.create_index(
        "ix_conversation_documents_document",
        "conversation_documents",
        ["document_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_documents_document",
        table_name="conversation_documents",
    )
    op.drop_table("conversation_documents")
    op.drop_column("messages", "sources")

"""Tables for the knowledge base: chunks (text, vector, full-text) and service areas."""
from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import get_settings
from app.core.db import Base


class KbChunk(Base):
    """One retrievable piece of the KB."""

    __tablename__ = "kb_chunks"

    chunk_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    section_no: Mapped[int] = mapped_column(Integer)
    section_title: Mapped[str] = mapped_column(Text)
    page: Mapped[int] = mapped_column(Integer)
    chunk_type: Mapped[str] = mapped_column(String(20))  # prose | table_row | faq | policy
    content: Mapped[str] = mapped_column(Text)
    # "metadata" is reserved on declarative models, so the attribute is `meta`.
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    embedding = mapped_column(Vector(get_settings().embedding_dim))
    doc_version: Mapped[str] = mapped_column(String(12))
    # 'english' is written out because Postgres rejects the one-argument form in a generated column.
    tsv = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', section_title || ' ' || content)", persisted=True),
    )

    __table_args__ = (Index("ix_kb_chunks_tsv", "tsv", postgresql_using="gin"),)


class ServiceArea(Base):
    """One (city, region) row from KB Section 10. Noida and Ghaziabad appear in two regions."""

    __tablename__ = "service_areas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city: Mapped[str] = mapped_column(String(80))
    region: Mapped[str] = mapped_column(String(80))
    source: Mapped[str] = mapped_column(String(40))

    __table_args__ = (UniqueConstraint("city", "region", name="uq_service_areas_city_region"),)

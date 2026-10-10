"""KB ingestion: PDF -> sections -> chunks -> embeddings -> database.

Usage:
    python -m app.rag.ingest --dry-run   parse and print, no database and no API key needed
    python -m app.rag.ingest             embed everything, then replace the KB rows in one transaction
"""
import argparse
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import check_db, get_engine, init_db
from app.core.llm import EmbeddingError
from app.core.llm import embedder as get_embedder
from app.rag.chunker import Chunk, chunk_sections
from app.rag.models import KbChunk, ServiceArea
from app.rag.pdf_parser import Section, parse_kb
from app.services.service_areas import SOURCE as SERVICE_AREA_SOURCE
from app.services.service_areas import ServiceAreas

CHUNK_TYPES = ("prose", "table_row", "faq", "policy")


@dataclass
class IngestResult:
    """What a full ingest wrote."""

    chunks: list[Chunk]
    service_area_rows: int
    doc_version: str
    model: str
    dim: int


def print_summary(sections: list[Section], chunks: list[Chunk]) -> None:
    """Print chunk counts per section and type, then the totals."""
    counts = Counter((c.section_no, c.chunk_type) for c in chunks)
    print(f"{'Sec':>3}  {'Page':>4}  {'Title':<40}" + "".join(f"{t:>10}" for t in CHUNK_TYPES))
    for s in sections:
        row = "".join(f"{counts[(s.number, t)]:>10}" for t in CHUNK_TYPES)
        print(f"{s.number:>3}  {s.page:>4}  {s.title:<40}{row}")
    totals = Counter(c.chunk_type for c in chunks)
    print(f"{'':>3}  {'':>4}  {'TOTAL':<40}" + "".join(f"{totals[t]:>10}" for t in CHUNK_TYPES))
    print(f"\n{len(sections)} sections, {len(chunks)} chunks\n")


def print_chunks(chunks: list[Chunk]) -> None:
    """Print each chunk's id and the first 80 characters of its content."""
    for c in chunks:
        preview = c.content.replace("\n", " / ")[:80]
        print(f"{c.chunk_id:<52} {c.chunk_type:<9} {preview}")


def doc_version_of(path: Path) -> str:
    """First 12 hex characters of the PDF's sha256, to tell KB versions apart."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def write_kb(
    engine: Engine, records: list[KbChunk], area_rows: list[tuple[str, str]]
) -> None:
    """Replace all kb_chunks and service_areas rows in ONE transaction (all or nothing)."""
    with Session(engine) as session, session.begin():
        session.execute(delete(KbChunk))
        session.execute(delete(ServiceArea))
        session.add_all(records)
        session.add_all(
            ServiceArea(city=city, region=region, source=SERVICE_AREA_SOURCE)
            for city, region in area_rows
        )


def ingest_sections(
    sections: list[Section], doc_version: str, engine: Engine, embedder
) -> IngestResult:
    """Chunk and embed everything first, then write. If embedding fails nothing is written.

    Uses only the engine and embedder passed in, so tests can point it at the test database.
    """
    chunks = chunk_sections(sections)
    area_rows = ServiceAreas.from_sections(sections).rows()
    vectors = embedder.embed_documents([c.embed_text() for c in chunks])  # raises on any failure
    if len(vectors) != len(chunks):
        raise EmbeddingError(f"expected {len(chunks)} vectors, got {len(vectors)}")
    records = [
        KbChunk(
            chunk_id=c.chunk_id,
            section_no=c.section_no,
            section_title=c.section_title,
            page=c.page,
            chunk_type=c.chunk_type,
            content=c.content,
            meta={**c.metadata, "embedding_model": embedder.model},
            embedding=vector,
            doc_version=doc_version,
        )
        for c, vector in zip(chunks, vectors)
    ]
    init_db(engine)  # creates the tables if they are missing
    write_kb(engine, records, area_rows)
    return IngestResult(chunks, len(area_rows), doc_version, embedder.model, embedder.dim)


def print_result(result: IngestResult) -> None:
    """Print a short summary of what was written."""
    counts = Counter(c.chunk_type for c in result.chunks)
    by_type = ", ".join(f"{t} {counts[t]}" for t in CHUNK_TYPES)
    print(f"Wrote {len(result.chunks)} chunks ({by_type})")
    print(f"Wrote {result.service_area_rows} service-area rows")
    print(f"doc_version {result.doc_version}, model {result.model}, dim {result.dim}")


def _short(exc: Exception) -> str:
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0][:200] if lines else ''}"


def main() -> int:
    """Command-line entry point. Returns the exit code."""
    sys.stdout.reconfigure(encoding="utf-8")  # the KB has bullets and arrows
    parser = argparse.ArgumentParser(description="Ingest the Vantis KB PDF.")
    parser.add_argument("--dry-run", action="store_true", help="parse and print, write nothing")
    args = parser.parse_args()

    kb_file = get_settings().kb_file
    if not kb_file.exists():
        print(
            f"KB PDF not found at {kb_file}.\n"
            "Put your copy at data/kb/Vantis_Energy_Solutions_Knowledge_Base.pdf "
            "(or set KB_PATH in backend/.env)."
        )
        return 1

    sections = parse_kb(kb_file)
    if args.dry_run:
        chunks = chunk_sections(sections)
        print_summary(sections, chunks)
        print_chunks(chunks)
        return 0

    # Check the database first, so a bad connection does not waste embedding quota.
    engine = get_engine()
    if engine is None:
        print("DATABASE_URL is not set in backend/.env.")
        return 1
    if not check_db(engine):
        print("Cannot reach the database. Check DATABASE_URL (Neon may be waking up; retry).")
        return 1
    try:
        result = ingest_sections(sections, doc_version_of(kb_file), engine, get_embedder())
    except EmbeddingError as exc:
        print(f"Embedding failed, the database was not changed.\n{_short(exc)}")
        return 1
    except Exception as exc:
        print(f"Database write failed, nothing was changed.\n{_short(exc)}")
        return 1
    print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

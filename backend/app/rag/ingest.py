"""KB ingestion. For now only the dry run exists (T2); database writes come in T3.

Usage: python -m app.rag.ingest --dry-run
"""
import argparse
import sys
from collections import Counter

from app.core.config import get_settings
from app.rag.chunker import Chunk, chunk_sections
from app.rag.pdf_parser import Section, parse_kb

CHUNK_TYPES = ("prose", "table_row", "faq", "policy")


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
    if not args.dry_run:
        print("Writing to the database comes in T3. Use --dry-run for now.")
        return 2

    sections = parse_kb(kb_file)
    chunks = chunk_sections(sections)
    print_summary(sections, chunks)
    print_chunks(chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())

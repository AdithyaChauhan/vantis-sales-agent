"""Search the KB from the command line and show how each chunk ranked.

Run: python -m scripts.search_kb "Is there a guarantee on the panels?" [--mode keyword|vector|hybrid]
"""
import argparse
import sys

from app.rag.retriever import RetrievedChunk, search


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_hit(position: int, c: RetrievedChunk) -> None:
    """Print one chunk: where it is from, its scores and ranks, and a short preview."""
    print(f"{position}. [KB Section {c.section_no}: {c.section_title}] page {c.page} ({c.chunk_type}) {c.chunk_id}")
    print(
        f"   rrf {_fmt(c.rrf_score, 4)} | keyword rank {c.keyword_rank or '-'} "
        f"score {_fmt(c.keyword_score)} | vector rank {c.vector_rank or '-'} "
        f"similarity {_fmt(c.similarity)}"
    )
    print(f"   {c.content.replace(chr(10), ' / ')[:80]}")


def main() -> int:
    """Command-line entry point. Returns the exit code."""
    sys.stdout.reconfigure(encoding="utf-8")  # the KB has bullets and arrows
    parser = argparse.ArgumentParser(description="Search the Vantis KB.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    result = search(args.question, k=args.k, mode=args.mode)
    print(f"mode: {result.mode} | no_context: {result.no_context} | degraded: {result.degraded}")
    for position, chunk in enumerate(result.chunks, start=1):
        print_hit(position, chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Retrieval evaluation: hit@1, hit@3 and MRR for keyword, vector and hybrid search, plus the relevance gate.

Run: python -m scripts.eval_retrieval [--rewrite]

Reads data/eval/retrieval_eval.jsonl (KB questions with the sections that answer them) and
data/eval/offtopic_eval.jsonl (questions the KB cannot answer). Each question is embedded once
(about 32 API calls) and that vector is reused for every mode. Prints the report and writes
docs/EVAL.md; the hand-written "Tuning notes" section of that file is kept.

--rewrite also runs the agent's understand step on every question (one fast_llm call each, paced to stay
under the free tier's 15 requests a minute), embeds the rewritten search_query and adds two modes:
"rewrite" (hybrid search on the rewritten query) and "fused" (RRF of the raw-question and rewritten
hybrid rankings). It also reports where the small-to-big rule (app/agent/graph.py) would add a table
summary chunk to the agent's 4-chunk context.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from sqlalchemy import Engine, text

from app.core.config import REPO_ROOT, get_settings
from app.core.db import check_db, get_engine
from app.core.llm import EmbeddingError, LLMError, chat_llm
from app.core.llm import embedder as get_embedder
from app.rag.retriever import (
    COMMON_WORD_RATIO,
    KEYWORD_WEIGHT,
    RRF_K,
    SearchResult,
    count_kb_chunks,
    search,
)

KB_QUESTIONS = REPO_ROOT / "data" / "eval" / "retrieval_eval.jsonl"
OFFTOPIC_QUESTIONS = REPO_ROOT / "data" / "eval" / "offtopic_eval.jsonl"
REPORT_PATH = REPO_ROOT / "docs" / "EVAL.md"
NOTES_HEADING = "## Tuning notes"
NOTES_PLACEHOLDER = (
    f"{NOTES_HEADING}\n\n"
    "(Write the tuning notes here. The script keeps this heading and everything below it.)\n"
)
TOP_K = 10  # chunks requested from each search
MODES = ("keyword", "vector", "hybrid")
REWRITE_MODES = ("rewrite", "fused")
CONTEXT_K = 4  # chunks the agent gets
REWRITE_PAUSE_S = 4.5  # about 13 understand calls a minute (free tier: 15)

_KB_INFO_SQL = text(
    """
    SELECT doc_version, metadata->>'embedding_model' AS model,
           count(*) AS chunks, count(*) FILTER (WHERE chunk_type <> 'policy') AS searchable
    FROM kb_chunks
    GROUP BY doc_version, metadata->>'embedding_model'
    """
)


class EvalSetupError(RuntimeError):
    """Something is missing or inconsistent, so the eval cannot give trustworthy numbers."""


@dataclass
class Question:
    """One eval question. `expected` holds the KB section numbers that count as a hit (empty = off-topic)."""

    id: str
    question: str
    expected: set[int]
    note: str = ""


@dataclass
class Evaluated:
    """A question together with the result of each search mode."""

    question: Question
    results: dict[str, SearchResult]

    def first_hit(self, mode: str) -> int | None:
        """Position of the first returned chunk from an expected section, or None."""
        sections = [c.section_no for c in self.results[mode].chunks]
        return first_hit_rank(sections, self.question.expected)

    @property
    def best_similarity(self) -> float | None:
        """Best cosine similarity of the question to any chunk (the vector search's first hit)."""
        chunks = self.results["vector"].chunks
        return chunks[0].similarity if chunks else None

    @property
    def keyword_hits(self) -> int:
        """How many chunks the keyword search found (at most TOP_K)."""
        return len(self.results["keyword"].chunks)


class FixedEmbedder:
    """Hands search() the vectors computed up front, so the eval embeds each question only once."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


# ---------- metrics ----------


def first_hit_rank(sections: list[int], expected: set[int]) -> int | None:
    """1-based position of the first returned section that is expected, or None if there is none."""
    for position, section in enumerate(sections, start=1):
        if section in expected:
            return position
    return None


def count_hits(ranks: list[int | None], k: int) -> int:
    """How many questions have their first hit within the top k."""
    return sum(1 for rank in ranks if rank is not None and rank <= k)


def hit_rate(ranks: list[int | None], k: int) -> float:
    """Share of questions with a hit in the top k (hit@k)."""
    return count_hits(ranks, k) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank. A question with no hit counts as 0."""
    return sum(1 / rank for rank in ranks if rank) / len(ranks) if ranks else 0.0


# ---------- loading and running ----------


def load_questions(path: Path) -> list[Question]:
    """Read a JSONL file. Whitespace is collapsed like search() does; off-topic rows have no sections."""
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            Question(
                id=row["id"],
                question=" ".join(row["question"].split()),
                expected={int(s) for s in row.get("expected_sections", [])},
                note=row.get("note", ""),
            )
        )
    return questions


def read_kb_info(engine: Engine | None, embedding_model: str) -> dict:
    """doc_version, model and chunk counts of the ingested KB. Raises EvalSetupError if it cannot be used."""
    if engine is None:
        raise EvalSetupError("DATABASE_URL is not set in backend/.env.")
    if not check_db(engine):
        raise EvalSetupError("cannot reach the database (Neon may be waking up; retry).")
    if not count_kb_chunks(engine):
        raise EvalSetupError("kb_chunks is empty; run python -m app.rag.ingest first.")
    with engine.connect() as conn:
        rows = conn.execute(_KB_INFO_SQL).mappings().all()
    if len(rows) != 1:
        raise EvalSetupError("kb_chunks mixes several doc_versions or embedding models; re-run ingest.")
    info = dict(rows[0])
    if info["model"] != embedding_model:
        raise EvalSetupError(
            f"the KB was embedded with {info['model']} but EMBEDDING_MODEL is {embedding_model}; "
            "vectors from different models cannot be compared. Re-run ingest."
        )
    return info


def embed_questions(questions: list[Question], embedder) -> FixedEmbedder:
    """Embed every question once. Raises EmbeddingError if the API fails."""
    return FixedEmbedder({q.question: embedder.embed_query(q.question) for q in questions})


def evaluate(
    questions: list[Question],
    engine: Engine,
    embedder: FixedEmbedder,
    rewrites: dict[str, str | None] | None = None,
) -> list[Evaluated]:
    """Run keyword, vector and hybrid search (top TOP_K) for every question.

    With rewrites (question -> rewritten search_query, or None), also the "rewrite" and "fused" modes.
    """
    rows = []
    for q in questions:
        results = {}
        for mode in MODES:
            result = search(q.question, k=TOP_K, mode=mode, engine=engine, embedder=embedder)
            if result.degraded:  # would mean vector numbers were really keyword numbers
                raise EvalSetupError(f"{q.id}: the {mode} search fell back to keyword-only.")
            results[mode] = result
        if rewrites is not None:
            query = rewrites.get(q.question)
            if query:
                results["rewrite"] = search(query, k=TOP_K, mode="hybrid", engine=engine, embedder=embedder)
            else:  # the understand step found nothing to search for
                results["rewrite"] = SearchResult(mode="hybrid", no_context=True)
            results["fused"] = fuse_results(results["hybrid"], results["rewrite"])
        rows.append(Evaluated(q, results))
    return rows


def fuse_results(raw: SearchResult, rewritten: SearchResult, k: int = RRF_K) -> SearchResult:
    """RRF of two rankings with equal weight (raw-question hybrid and rewritten-query hybrid).

    no_context only when both searches say so.
    """
    scores: dict[str, float] = {}
    chunks = {}
    for ranking in (raw.chunks, rewritten.chunks):
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (k + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return SearchResult(
        chunks=[chunks[cid] for cid in order][:TOP_K],
        mode="fused",
        no_context=raw.no_context and rewritten.no_context,
    )


def rewrite_questions(questions: list[Question], llm, hints, pause_s: float = REWRITE_PAUSE_S) -> dict:
    """question -> the understand step's search_query (None when it found nothing to search for).

    A failed call keeps the raw question, as the agent does. Paced to respect the per-minute quota.
    """
    from app.agent.understand import understand

    rewrites: dict[str, str | None] = {}
    for i, q in enumerate(questions):
        if i:
            time.sleep(pause_s)
        try:
            rewrites[q.question] = understand(llm, hints, q.question, [])["search_query"]
        except LLMError as exc:
            print(f"  {q.id}: understand failed ({exc}); using the raw question")
            rewrites[q.question] = q.question
    return rewrites


def table_complete(chunks: list[dict], section: int, row_counts: dict[int, int]) -> bool:
    """True if the context holds the whole table of a section: its summary chunk, or every row chunk."""
    if any(c["section_no"] == section and c["chunk_type"] == "table_summary" for c in chunks):
        return True
    rows = sum(1 for c in chunks if c["section_no"] == section and c["chunk_type"] == "table_row")
    return rows > 0 and rows == row_counts.get(section, -1)


# ---------- report ----------


def report_modes(rows: list[Evaluated]) -> tuple[str, ...]:
    """The modes present in the results (the rewrite modes only after --rewrite)."""
    return MODES + tuple(m for m in REWRITE_MODES if rows and m in rows[0].results)


def metrics_lines(rows: list[Evaluated]) -> list[str]:
    """Table of hit@1, hit@3 and MRR per mode."""
    lines = ["| Mode | hit@1 | hit@3 | MRR |", "|---|---|---|---|"]
    for mode in report_modes(rows):
        ranks = [row.first_hit(mode) for row in rows]
        cells = [f"{hit_rate(ranks, k):.3f} ({count_hits(ranks, k)}/{len(ranks)})" for k in (1, 3)]
        lines.append(f"| {mode} | {cells[0]} | {cells[1]} | {mrr(ranks):.3f} |")
    return lines


def misses_lines(rows: list[Evaluated]) -> list[str]:
    """Table of questions with no expected section in the top 3, for each mode."""
    lines = [
        "| Mode | Question | Expected sections | Top 3 returned | First hit at |",
        "|---|---|---|---|---|",
    ]
    for mode in report_modes(rows):
        for row in rows:
            rank = row.first_hit(mode)
            if rank is not None and rank <= 3:
                continue
            expected = ", ".join(str(s) for s in sorted(row.question.expected))
            top3 = ", ".join(c.chunk_id for c in row.results[mode].chunks[:3]) or "(nothing)"
            question = f"{row.question.id}: {row.question.question}"
            lines.append(f"| {mode} | {question} | {expected} | {top3} | {rank or '-'} |")
    return lines


def _no_context_count(rows: list[Evaluated], mode: str) -> int:
    return sum(1 for row in rows if row.results[mode].no_context)


def _fmt_similarity(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _similarity_range(rows: list[Evaluated]) -> str:
    sims = sorted((row.best_similarity, row.question.id) for row in rows if row.best_similarity is not None)
    if not sims:
        return "-"
    (low, low_id), (high, high_id) = sims[0], sims[-1]
    return f"{low:.3f} ({low_id}) to {high:.3f} ({high_id})"


def gate_lines(kb_rows: list[Evaluated], off_rows: list[Evaluated], threshold: float) -> list[str]:
    """Counts and per-question details for the relevance gate (no_context)."""
    kb_n, off_n = len(kb_rows), len(off_rows)
    lines = [
        f"Relevance threshold: {threshold} (best cosine similarity needed when a vector search ran).",
        "",
        "| Check | hybrid | keyword-only (degraded) | Target |",
        "|---|---|---|---|",
        f"| KB questions wrongly marked no_context | {_no_context_count(kb_rows, 'hybrid')} of {kb_n} "
        f"| {_no_context_count(kb_rows, 'keyword')} of {kb_n} | 0 |",
        f"| Off-topic questions marked no_context | {_no_context_count(off_rows, 'hybrid')} of {off_n} "
        f"| {_no_context_count(off_rows, 'keyword')} of {off_n} | {off_n} |",
        "",
        f"Best similarity, KB questions: {_similarity_range(kb_rows)}.",
        f"Best similarity, off-topic questions: {_similarity_range(off_rows)}.",
        "",
        "| Off-topic question | Best similarity | Keyword hits "
        "| no_context (hybrid) | no_context (keyword-only) |",
        "|---|---|---|---|---|",
    ]
    for row in off_rows:
        lines.append(
            f"| {row.question.id}: {row.question.question} | {_fmt_similarity(row.best_similarity)} "
            f"| {row.keyword_hits} | {row.results['hybrid'].no_context} "
            f"| {row.results['keyword'].no_context} |"
        )
    wrong = [row for row in kb_rows if row.results["hybrid"].no_context]
    if wrong:
        lines += ["", "| KB question wrongly marked no_context | Best similarity | Keyword hits |"]
        lines += ["|---|---|---|"]
        lines += [
            f"| {row.question.id}: {row.question.question} | {_fmt_similarity(row.best_similarity)} "
            f"| {row.keyword_hits} |"
            for row in wrong
        ]
    return lines


def rewrite_lines(kb_rows: list[Evaluated], off_rows: list[Evaluated], rewrites: dict) -> list[str]:
    """The rewritten queries, and the off-topic gate for the rewrite mode."""
    off_blocked = sum(1 for row in off_rows if row.results["rewrite"].no_context)
    kb_blocked = sum(1 for row in kb_rows if row.results["rewrite"].no_context)
    lines = [
        f"Off-topic questions marked no_context (rewrite mode): {off_blocked} of {len(off_rows)}. "
        f"KB questions wrongly marked no_context: {kb_blocked} of {len(kb_rows)}.",
        "",
        "| Question | Rewritten search_query | First hit (hybrid → rewrite) |",
        "|---|---|---|",
    ]
    for row in kb_rows + off_rows:
        query = rewrites.get(row.question.question) or "(none)"
        before = row.first_hit("hybrid") if row.question.expected else None
        after = row.first_hit("rewrite") if row.question.expected else None
        moved = f"{before or '-'} → {after or '-'}" if row.question.expected else "off-topic"
        lines.append(f"| {row.question.id}: {row.question.question} | {query} | {moved} |")
    return lines


def small_to_big_lines(kb_rows: list[Evaluated], summaries: dict[int, dict], row_counts: dict[int, int]) -> list[str]:
    """Where the small-to-big rule adds a table summary to the agent's 4-chunk context (rewrite mode)."""
    from app.agent.graph import add_table_summaries

    lines = ["| Question | Summary added | Full table before → after |", "|---|---|---|"]
    fired = 0
    for row in kb_rows:
        chunks = [asdict(c) for c in row.results["rewrite"].chunks[:CONTEXT_K]]
        after = add_table_summaries(chunks, summaries)
        added = [c["chunk_id"] for c in after[len(chunks):]]
        if not added:
            continue
        fired += 1
        sections = sorted({c["section_no"] for c in after if c["chunk_type"] == "table_summary"})
        before_ok = all(table_complete(chunks, s, row_counts) for s in sections)
        after_ok = all(table_complete(after, s, row_counts) for s in sections)
        lines.append(f"| {row.question.id}: {row.question.question} | {', '.join(added)} | {before_ok} → {after_ok} |")
    return [f"The rule fires for {fired} of {len(kb_rows)} KB questions.", "", *lines] if fired else [
        f"The rule fires for 0 of {len(kb_rows)} KB questions."
    ]


def build_report(
    kb_rows: list[Evaluated],
    off_rows: list[Evaluated],
    setup: list[tuple[str, str]],
    threshold: float,
    extra: list[str] | None = None,
) -> str:
    """The generated part of docs/EVAL.md, as markdown."""
    lines = [
        "# EVAL.md: retrieval evaluation",
        "",
        "Generated by `python -m scripts.eval_retrieval`. The script rewrites everything above "
        'the "Tuning notes" heading on every run; the notes below it are written by hand and kept.',
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        *[f"| {label} | {value} |" for label, value in setup],
        "",
        f"## Results ({len(kb_rows)} KB questions)",
        "",
        "A hit is a returned chunk whose section is one of the question's expected sections. "
        "hit@k is the share of questions with a hit in the first k results; MRR is the mean of 1 / (position "
        f"of the first hit), with 0 when there is no hit in the top {TOP_K}.",
        "",
        *metrics_lines(kb_rows),
        "",
        "## Misses (no expected section in the top 3)",
        "",
        *misses_lines(kb_rows),
        "",
        "## Relevance gate",
        "",
        *gate_lines(kb_rows, off_rows, threshold),
        *(extra or []),
    ]
    return "\n".join(lines)


def write_report(path: Path, generated: str) -> None:
    """Write the generated report above the hand-written notes, which are kept as they are."""
    notes = NOTES_PLACEHOLDER
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        start = existing.find("\n" + NOTES_HEADING)
        if start != -1:
            notes = existing[start + 1 :]
    path.write_text(generated.rstrip("\n") + "\n\n" + notes, encoding="utf-8", newline="\n")


def main() -> int:
    """Command-line entry point. Returns the exit code."""
    sys.stdout.reconfigure(encoding="utf-8")  # keeps a Windows console from choking on non-ASCII text
    parser = argparse.ArgumentParser(description="Retrieval eval; writes docs/EVAL.md.")
    parser.add_argument("--rewrite", action="store_true", help="also evaluate the understand step's rewrites")
    args = parser.parse_args()
    settings = get_settings()
    rewrites = None
    try:
        kb_questions = load_questions(KB_QUESTIONS)
        off_questions = load_questions(OFFTOPIC_QUESTIONS)
        questions = kb_questions + off_questions
        engine = get_engine()
        info = read_kb_info(engine, settings.embedding_model)
        print(f"Embedding {len(questions)} questions (one API call each)...")
        embedder = embed_questions(questions, get_embedder())
        if args.rewrite:
            from app.agent.understand import Understanding, load_kb_hints

            hints = load_kb_hints(engine)
            print(f"Rewriting {len(questions)} questions with the understand step (paced, about 2.5 minutes)...")
            rewrites = rewrite_questions(questions, chat_llm(settings.llm_model_fast, schema=Understanding), hints)
            new = [q for q in dict.fromkeys(rewrites.values()) if q and q not in embedder.vectors]
            print(f"Embedding {len(new)} rewritten queries...")
            embedder.vectors.update({q: get_embedder().embed_query(q) for q in new})
        rows = evaluate(questions, engine, embedder, rewrites)
    except EmbeddingError as exc:
        print(f"Embedding failed, nothing was written.\n{exc}")
        return 1
    except (EvalSetupError, OSError) as exc:
        print(f"Cannot run the eval: {exc}")
        return 1

    policy_chunks = info["chunks"] - info["searchable"]
    setup = [
        ("Date", date.today().isoformat()),
        ("Embedding model", settings.embedding_model),
        ("Dimensions", str(settings.embedding_dim)),
        ("KB doc_version", info["doc_version"]),
        ("Chunks", f"{info['chunks']} ({info['searchable']} searchable, {policy_chunks} policy)"),
        ("Questions", f"{len(kb_questions)} KB, {len(off_questions)} off-topic"),
        ("Results per search", str(TOP_K)),
        ("RRF k", str(RRF_K)),
        ("Keyword weight in RRF", f"{KEYWORD_WEIGHT} (vector list = 1)"),
        ("Common-word cutoff", f"query words in more than {COMMON_WORD_RATIO:.0%} of chunks are dropped"),
        ("Relevance threshold", f"{settings.retrieval_min_similarity} (best cosine similarity)"),
    ]
    split = len(kb_questions)
    extra = None
    if rewrites is not None:
        with engine.connect() as conn:
            row_counts = dict(conn.execute(text(
                "SELECT section_no, count(*) FROM kb_chunks WHERE chunk_type = 'table_row' GROUP BY section_no"
            )).all())
        setup.append(("Rewrite", f"understand step ({settings.llm_model_fast}), no history; fused = RRF of "
                                 "raw and rewritten hybrid rankings"))
        extra = [
            "", "## Query rewrite (understand step)", "",
            *rewrite_lines(rows[:split], rows[split:], rewrites),
            "", "## Small-to-big (rewrite mode, the agent's 4-chunk context)", "",
            *small_to_big_lines(rows[:split], hints.summaries, row_counts),
        ]
    report = build_report(rows[:split], rows[split:], setup, settings.retrieval_min_similarity, extra)
    print(report)
    write_report(REPORT_PATH, report)
    print(f"\nWrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

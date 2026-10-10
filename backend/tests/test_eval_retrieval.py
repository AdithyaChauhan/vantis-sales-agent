"""Eval script helpers on tiny fixtures. No database and no network."""
import json

import pytest

from app.rag.retriever import RetrievedChunk, SearchResult
from scripts.eval_retrieval import (
    KB_QUESTIONS,
    NOTES_HEADING,
    OFFTOPIC_QUESTIONS,
    Evaluated,
    Question,
    build_report,
    count_hits,
    first_hit_rank,
    gate_lines,
    hit_rate,
    load_questions,
    mrr,
    write_report,
)


def chunk(section, similarity=None):
    return RetrievedChunk(f"s{section:02d}-x", section, "Title", 1, "prose", "content", similarity=similarity)


def result(*sections, no_context=False):
    return SearchResult(chunks=[chunk(s, 0.7) for s in sections], no_context=no_context)


def evaluated(qid, expected, keyword=(), vector=(), hybrid=(), no_context=(False, False)):
    """An Evaluated question; no_context is (hybrid, keyword-only)."""
    results = {
        "keyword": result(*keyword, no_context=no_context[1]),
        "vector": result(*vector),
        "hybrid": result(*hybrid, no_context=no_context[0]),
    }
    return Evaluated(Question(qid, f"question {qid}", set(expected)), results)


# ---- metrics ----


def test_first_hit_rank():
    assert first_hit_rank([6, 8, 10], {8}) == 2
    assert first_hit_rank([6, 8, 10], {10, 8}) == 2  # the earliest expected section wins
    assert first_hit_rank([6, 1], {8}) is None
    assert first_hit_rank([], {8}) is None


def test_hit_rate_and_count_hits():
    ranks = [1, 3, None, 4]
    assert count_hits(ranks, 1) == 1
    assert count_hits(ranks, 3) == 2
    assert hit_rate(ranks, 1) == 0.25
    assert hit_rate(ranks, 3) == 0.5
    assert hit_rate(ranks, 10) == 0.75  # a miss never counts
    assert hit_rate([], 3) == 0.0


def test_mrr():
    assert mrr([1, 2, None, 4]) == pytest.approx((1 + 1 / 2 + 0 + 1 / 4) / 4)
    assert mrr([None, None]) == 0.0
    assert mrr([1]) == 1.0
    assert mrr([]) == 0.0


# ---- loading ----


def test_load_questions(tmp_path):
    path = tmp_path / "q.jsonl"
    lines = [
        {"id": "q1", "question": "What  is   it?", "expected_sections": ["8", "9"], "note": "n"},
        {"id": "o1", "question": "Tell me a joke."},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n\n", encoding="utf-8")
    first, second = load_questions(path)
    assert first.question == "What is it?"  # whitespace collapsed, like search() does
    assert first.expected == {8, 9} and first.note == "n"
    assert second.expected == set()  # off-topic questions have no expected sections


def test_the_real_eval_files():
    kb = load_questions(KB_QUESTIONS)
    off = load_questions(OFFTOPIC_QUESTIONS)
    assert len(kb) == 24 and len(off) == 8
    assert all(q.expected and all(1 <= s <= 15 for s in q.expected) for q in kb)  # never policy sections
    assert all(not q.expected for q in off)
    assert len({q.id for q in kb + off}) == 32


# ---- the report ----


def test_evaluated_helpers():
    row = evaluated("q1", {8}, keyword=[6, 8], vector=[8, 6])
    assert row.first_hit("keyword") == 2
    assert row.first_hit("vector") == 1
    assert row.first_hit("hybrid") is None
    assert row.best_similarity == 0.7
    assert row.keyword_hits == 2


def test_report_has_the_numbers_and_the_misses():
    rows = [
        evaluated("q1", {8}, keyword=[8], vector=[8], hybrid=[8]),  # hit at 1 everywhere
        evaluated("q2", {10}, keyword=[1, 2, 3, 10], vector=[2, 10], hybrid=[2, 10]),  # keyword misses at 3
    ]
    off = [evaluated("o1", set(), no_context=(True, False))]
    report = build_report(rows, off, [("Date", "2026-10-11")], 0.63)
    assert "| keyword | 0.500 (1/2) | 0.500 (1/2) | 0.625 |" in report  # ranks 1 and 4: MRR (1 + 1/4) / 2
    assert "| vector | 0.500 (1/2) | 1.000 (2/2) | 0.750 |" in report
    assert "| hybrid | 0.500 (1/2) | 1.000 (2/2) | 0.750 |" in report
    assert "| keyword | q2: question q2 | 10 |" in report  # the keyword miss is listed ...
    assert "| vector | q2" not in report and "| hybrid | q2" not in report  # ... the others are not
    assert "| Date | 2026-10-11 |" in report
    assert chr(167) not in report  # the project writes "Section N", never the section sign (U+00A7)


def test_gate_counts_wrong_and_right_no_context():
    kb = [evaluated("q1", {8}, no_context=(False, False)), evaluated("q2", {8}, no_context=(True, False))]
    off = [evaluated("o1", set(), no_context=(True, True)), evaluated("o2", set(), no_context=(False, True))]
    text = "\n".join(gate_lines(kb, off, 0.63))
    assert "| Check | hybrid | keyword-only (degraded) | Target |" in text
    assert "| KB questions wrongly marked no_context | 1 of 2 | 0 of 2 | 0 |" in text
    assert "| Off-topic questions marked no_context | 1 of 2 | 2 of 2 | 2 |" in text
    assert "| q2: question q2 |" in text  # the wrongly blocked KB question is listed


# ---- writing docs/EVAL.md ----


def test_write_report_creates_the_file_with_a_notes_placeholder(tmp_path):
    path = tmp_path / "EVAL.md"
    write_report(path, "# Report\n")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Report\n")
    assert NOTES_HEADING in text


def test_write_report_keeps_the_hand_written_notes(tmp_path):
    path = tmp_path / "EVAL.md"
    path.write_text(f"# Old report\n\nold numbers\n\n{NOTES_HEADING}\n\nmy careful notes\n", encoding="utf-8")
    write_report(path, "# New report\n\nnew numbers")
    text = path.read_text(encoding="utf-8")
    assert "new numbers" in text and "old numbers" not in text
    assert text.endswith(f"{NOTES_HEADING}\n\nmy careful notes\n")
    write_report(path, "# Newer report")  # a second run still keeps the notes
    assert path.read_text(encoding="utf-8").endswith("my careful notes\n")
    assert "\r" not in path.read_text(encoding="utf-8")  # LF line endings

"""Parsing and chunking of the real KB PDF. Skipped when the PDF is missing."""
import re

import pytest

from app.core.config import get_settings
from app.rag.chunker import chunk_sections
from app.rag.pdf_parser import parse_kb, table_dicts

pytestmark = pytest.mark.skipif(not get_settings().kb_file.exists(), reason="KB PDF not found")

OFFERINGS = [
    "Vantis HomeStart 3",
    "Vantis HomePro 5",
    "Vantis HomeMax 10",
    "Vantis Business 15",
    "Vantis Business 25",
    "Vantis Site Assessment",
    "Vantis Care",
]


@pytest.fixture(scope="module")
def sections():
    return parse_kb(get_settings().kb_file)


@pytest.fixture(scope="module")
def chunks(sections):
    return chunk_sections(sections)


def rows(sections, number):
    section = next(s for s in sections if s.number == number)
    return [row for table in section.tables for row in table_dicts(table)]


def test_all_19_sections_found(sections):
    assert [s.number for s in sections] == list(range(1, 20))
    assert sections[7].title == "Product & Service Catalog"


def test_section_pages(sections):
    assert sections[0].page == 2  # cover page skipped
    section10 = sections[9]
    assert section10.page == 3
    assert any(line.page == 4 for line in section10.lines)  # continues onto page 4


def test_table_row_counts(sections):
    assert [r["Offering"] for r in rows(sections, 8)] == OFFERINGS
    assert len(rows(sections, 9)) == 5
    assert len(rows(sections, 10)) == 5
    assert len(rows(sections, 11)) == 5
    assert len(rows(sections, 12)) == 4


def test_cells_have_no_line_breaks(sections):
    for number in range(8, 13):
        for row in rows(sections, number):
            assert all("\n" not in v and "  " not in v for v in row.values())
    assert rows(sections, 8)[5]["Offering"] == "Vantis Site Assessment"


def test_faq_has_11_question_answer_chunks(chunks):
    faq = [c for c in chunks if c.chunk_type == "faq"]
    assert len(faq) == 11
    assert all(c.content.startswith("Q: ") and "\nA: " in c.content for c in faq)
    assert all(c.section_no == 15 for c in faq)


def test_sections_16_to_19_are_policy(chunks):
    policy_sections = {c.section_no for c in chunks if c.chunk_type == "policy"}
    assert policy_sections == {16, 17, 18, 19}
    assert all(c.chunk_type == "policy" for c in chunks if c.section_no >= 16)
    assert all(c.chunk_type != "policy" for c in chunks if c.section_no < 16)


def test_table_sections_have_one_chunk_per_row(chunks):
    counts = {n: sum(1 for c in chunks if c.section_no == n and c.chunk_type == "table_row") for n in range(8, 13)}
    assert counts == {8: 7, 9: 5, 10: 5, 11: 5, 12: 4}


def test_summary_chunks_for_sections_8_and_10(sections, chunks):
    summaries = {c.chunk_id: c for c in chunks if c.chunk_type == "table_summary"}
    assert sorted(summaries) == ["s08-table-summary", "s10-table-summary"]
    assert all(c.page == 3 for c in summaries.values())
    catalog = summaries["s08-table-summary"].content
    assert all(name in catalog for name in OFFERINGS)
    locations = summaries["s10-table-summary"].content
    for row in rows(sections, 10):
        for city in row["Reference Locations"].split(", "):
            assert city in locations


def test_no_page_header_or_footer_in_chunks(chunks):
    for c in chunks:
        assert "Participant Knowledge Base" not in c.content
        assert "Round 2 Technical Assignment" not in c.content
        assert "Pradrix" not in c.content
        assert "(cid:" not in c.content


def test_chunk_ids_unique_and_stable(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert "s08-vantis-homepro-5" in ids
    assert "s10-uttar-pradesh" in ids


def test_embed_text_has_header(chunks):
    chunk = next(c for c in chunks if c.chunk_id == "s08-vantis-homepro-5")
    text = chunk.embed_text()
    assert text.startswith(
        "title: KB Section 8: Product & Service Catalog - Vantis HomePro 5 | text: "
    )
    assert "5 kW rooftop solution" in text


def test_embed_title_has_no_pipe(chunks):
    for chunk in chunks:
        assert "|" not in chunk.embed_title()


def test_prose_paragraphs_are_joined(chunks):
    overview = next(c for c in chunks if c.chunk_id == "s01-company-overview")
    assert len(overview.content.split("\n")) == 3  # three paragraphs, wrapped lines joined


def by_id(chunks, chunk_id):
    return next(c for c in chunks if c.chunk_id == chunk_id)


def test_no_standalone_fi_word(chunks):
    for c in chunks:
        assert not re.search(r"\bfi\b", c.content), c.chunk_id


def test_arrows_and_words_are_intact(chunks):
    assert "→" in by_id(chunks, "s06-installation-site-assessment").content
    assert "→" in by_id(chunks, "s18-source-of-truth-rule").content
    all_text = " ".join(c.content for c in chunks)
    assert "specifications" in all_text
    assert "financing" in all_text


def test_section_8_prose_has_catalog_rule(chunks):
    assert "Important Catalog Rule" in by_id(chunks, "s08-product-service-catalog").content


def test_section_16_keeps_last_bullet_from_page_6(chunks):
    content = by_id(chunks, "s16-information-the-ai-must-not-invent").content
    assert "• Human representative actions that did not occur" in content

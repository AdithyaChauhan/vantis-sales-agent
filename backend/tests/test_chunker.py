"""Chunker rules on small synthetic sections (no PDF needed)."""
from app.rag.chunker import MAX_PROSE_CHARS, chunk_sections, slugify
from app.rag.pdf_parser import Line, Section, Table


def section(number, title, texts):
    lines = [Line(t, 2, True) for t in texts]
    return Section(number, title, 2, lines=lines)


def test_slugify():
    assert slugify("Vantis HomePro 5") == "vantis-homepro-5"
    assert slugify("Product & Service Catalog") == "product-service-catalog"
    long = slugify("How much does solar installation cost for a very large home in the city?")
    assert len(long) <= 50 and not long.endswith("-")


def test_long_prose_splits_at_paragraph_breaks():
    paragraph = "x" * 500
    chunks = chunk_sections([section(3, "Solar Solutions", [paragraph] * 4)])
    assert [c.chunk_id for c in chunks] == ["s03-solar-solutions", "s03-solar-solutions-2"]
    assert all(len(c.content) <= MAX_PROSE_CHARS for c in chunks)
    assert chunks[0].content == "\n".join([paragraph] * 2)


def test_short_prose_is_one_chunk():
    chunks = chunk_sections([section(5, "Pricing", ["One.", "Two."])])
    assert len(chunks) == 1 and chunks[0].chunk_type == "prose"


def test_wrapped_lines_join_but_bullets_stay_separate():
    lines = [
        Line("A sentence that", 2, True),
        Line("wraps here.", 2, False),
        Line("• first", 2, False),
        Line("• second", 2, False),
    ]
    chunk = chunk_sections([Section(14, "Info", 2, lines=lines)])[0]
    assert chunk.content == "A sentence that wraps here.\n• first\n• second"


def test_policy_tag():
    chunks = chunk_sections([section(16, "Must Not Invent", ["• Exact price"])])
    assert chunks[0].chunk_type == "policy"


def test_duplicate_ids_raise():
    import pytest

    twin = [section(3, "Same", ["a"]), section(3, "Same", ["b"])]
    with pytest.raises(ValueError):
        chunk_sections(twin)


def table_section(number, title):
    table = Table(3, ["Offering", "Customer"], [["Plan A", "Homes"], ["Plan B", "Shops"]])
    return Section(number, title, 3, tables=[table])


def test_summary_chunk_lists_every_row():
    chunks = chunk_sections([table_section(8, "Catalog")])
    summary = [c for c in chunks if c.chunk_type == "table_summary"]
    assert [c.chunk_id for c in summary] == ["s08-table-summary"]
    assert summary[0].content == "All rows (Offering; Customer):\n- Plan A; Homes\n- Plan B; Shops"
    assert summary[0].page == 3
    assert summary[0].metadata == {"label": "Summary of all rows", "rows": 2}
    assert [c.chunk_type for c in chunks].count("table_row") == 2  # the row chunks are still there


def test_other_table_sections_get_no_summary():
    chunks = chunk_sections([table_section(9, "Guidance")])
    assert "table_summary" not in [c.chunk_type for c in chunks]

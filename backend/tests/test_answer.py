"""Prompts, context format, citations and sources (the answer helpers the agent reuses)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.core.prompts import load_prompt
from app.rag.answer import (
    SNIPPET_CHARS,
    normalize_citations,
    rag_message,
    cited_sections,
    format_context,
    pick_sources,
    strip_tool_citations,
    tidy_citations,
)
from app.rag.retriever import RetrievedChunk


class FakeLLM:
    """Records every call. Returns a fixed reply, or raises `error`."""

    def __init__(self, reply="ok", provider="google_genai", error=None):
        self.reply, self.provider, self.error = reply, provider, error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.reply, response_metadata={"model_provider": self.provider})


def chunk(section_no, title, content="text", page=3, chunk_id=None):
    return RetrievedChunk(
        chunk_id=chunk_id or f"s{section_no:02d}-x",
        section_no=section_no,
        section_title=title,
        page=page,
        chunk_type="table_row",
        content=content,
    )


CATALOG = chunk(8, "Product & Service Catalog", "Offering: Vantis HomePro 5", chunk_id="s08-a")
CATALOG_2 = chunk(8, "Product & Service Catalog", "Offering: Vantis HomeMax 10", chunk_id="s08-b")
FAQ = chunk(15, "Frequently Asked Questions", "Q: Can I get a quote?\nA: Yes.", page=5)
CHUNKS = [CATALOG, FAQ, CATALOG_2]


def test_prompts_load_from_files():
    assert "Vantis" in load_prompt("system_prompt")
    rag = load_prompt("rag_prompt")
    assert "{context}" in rag and "{question}" in rag
    with pytest.raises(FileNotFoundError):
        load_prompt("no_such_prompt")


def test_context_blocks_start_with_their_citation_label():
    text = format_context([CATALOG, FAQ])
    assert text == (
        "[KB Section 8: Product & Service Catalog]\nOffering: Vantis HomePro 5\n\n"
        "[KB Section 15: Frequently Asked Questions]\nQ: Can I get a quote?\nA: Yes."
    )


def test_rag_message():
    message = rag_message("What is {context}?", [CATALOG])
    assert isinstance(message, HumanMessage)
    assert "[KB Section 8: Product & Service Catalog]\nOffering: Vantis HomePro 5" in message.content
    assert "What is {context}?" in message.content  # the customer's text is not expanded
    assert "{question}" not in message.content
    assert "<context>\n\n</context>" in rag_message("Hi", []).content  # empty context


def test_cited_sections():
    reply = "Yes [KB Section 8: Catalog]. Also [KB Section 15: FAQ] and again [KB Section 8: Catalog]."
    assert cited_sections(reply) == [8, 15]
    assert cited_sections("No citations here.") == []


def test_sources_are_the_cited_sections():
    sources = pick_sources("HomePro 5 [KB Section 8: Product & Service Catalog]", CHUNKS)
    assert sources == [{
        "section": "8",
        "title": "Product & Service Catalog",
        "page": 3,
        "snippet": "Offering: Vantis HomePro 5",
    }]


def test_one_source_per_section_in_retrieval_order():
    # 11, 11, 15, 11 -> 11, 15; the first (highest-ranked) section 11 chunk gives the snippet
    chunks = [
        chunk(11, "Warranty", "panels", chunk_id="s11-a"),
        chunk(11, "Warranty", "inverter", chunk_id="s11-b"),
        chunk(15, "FAQ", "faq", page=5),
        chunk(11, "Warranty", "work", chunk_id="s11-c"),
    ]
    sources = pick_sources("[KB Section 15: FAQ] and [KB Section 11: Warranty]", chunks)
    assert [(s["section"], s["snippet"]) for s in sources] == [("11", "panels"), ("15", "faq")]


def test_no_citation_means_all_context_sections():
    assert [s["section"] for s in pick_sources("No label here.", CHUNKS)] == ["8", "15"]
    assert len(pick_sources("Only [KB Section 99: Not In Context].", CHUNKS)) == 2


def test_snippet_is_one_line_and_cut_at_a_word():
    long_chunk = chunk(5, "Pricing", "word " * 100 + "\nend")
    [source] = pick_sources("[KB Section 5: Pricing]", [long_chunk])
    assert "\n" not in source["snippet"]
    assert len(source["snippet"]) <= SNIPPET_CHARS + 1 and source["snippet"].endswith("word…")


def test_full_width_citation_brackets_are_normalized():
    reply = "Steps: inquiry → handover【KB Section 8: Product & Service Catalog】."
    assert normalize_citations(reply) == "Steps: inquiry → handover[KB Section 8: Product & Service Catalog]."


TOOLS = ["search_knowledge_base", "check_service_area", "schedule_callback"]


@pytest.mark.parametrize("reply, expected", [
    ("Logged; the team will confirm it [schedule_callback result].", "Logged; the team will confirm it."),
    ("Not confirmed yet [check_service_area].", "Not confirmed yet."),
    ("Saved [create_lead result] and logged.", "Saved and logged."),
])
def test_tool_citations_are_removed(reply, expected):
    assert strip_tool_citations(reply, TOOLS + ["create_lead"]) == expected


def test_kb_labels_and_other_brackets_stay():
    reply = "Steps [KB Section 6: Installation & Site Assessment]. Results vary [KB Section 13: Results]. [note]"
    assert strip_tool_citations(reply, TOOLS) == reply
    assert strip_tool_citations("Plain [aside].", []) == "Plain [aside]."


AREAS_SUMMARY = chunk(10, "Reference Service Locations", "All rows", chunk_id="s10-table-summary")


def test_each_section_is_cited_once_at_its_last_place():
    label = "[KB Section 8: Product & Service Catalog]"
    reply = f"Options:\n1. A {label}\n2. B {label}\n3. C {label}\n\nWant me to check your city?"
    assert tidy_citations(reply, [CATALOG]) == f"Options:\n1. A\n2. B\n3. C {label}\n\nWant me to check your city?"


def test_a_bare_label_gets_its_title():
    assert tidy_citations("Vantis covers Lucknow [KB Section 10].", [AREAS_SUMMARY]) == (
        "Vantis covers Lucknow [KB Section 10: Reference Service Locations]."
    )
    assert tidy_citations("Unknown [KB Section 4].", [AREAS_SUMMARY]) == "Unknown [KB Section 4]."  # no title known


def test_clean_citations_are_left_alone():
    reply = ("Packages [KB Section 8: Product & Service Catalog]. "
             "Lucknow [KB Section 10: Reference Service Locations].")
    assert tidy_citations(reply, [CATALOG, AREAS_SUMMARY]) == reply

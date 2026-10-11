"""Prompts, context format, citations, sources and the answer flow, with a fake LLM."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.prompts import load_prompt
from app.rag.answer import (
    LLM_DOWN_REPLY,
    NO_CONTEXT_REPLY,
    SNIPPET_CHARS,
    answer,
    build_messages,
    cited_sections,
    format_context,
    pick_sources,
)
from app.rag.retriever import RetrievedChunk, SearchResult


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


def test_build_messages():
    system, user = build_messages("What is {context}?", [CATALOG])
    assert isinstance(system, SystemMessage) and system.content == load_prompt("system_prompt")
    assert isinstance(user, HumanMessage)
    assert "[KB Section 8: Product & Service Catalog]\nOffering: Vantis HomePro 5" in user.content
    assert "What is {context}?" in user.content  # the customer's text is not expanded
    assert "{question}" not in user.content


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


def test_answer_happy_path():
    llm = FakeLLM("We offer HomePro 5 [KB Section 8: Product & Service Catalog].")
    result = answer("What do you offer?", SearchResult(chunks=CHUNKS, mode="hybrid"), llm)
    assert result.reply.startswith("We offer HomePro 5")
    assert [s["section"] for s in result.sources] == ["8"]
    assert result.degraded is False
    retrieve, step = result.steps
    assert retrieve == {
        "node": "retrieve",
        "detail": "3 chunks (hybrid)",
        "data": {"mode": "hybrid", "chunks": 3, "no_context": False},
    }
    assert step["node"] == "answer" and step["data"]["provider"] == "gemini"
    assert isinstance(step["data"]["latency_ms"], int)
    assert len(llm.calls) == 1


def test_no_context_makes_no_llm_call():
    llm = FakeLLM()
    result = answer("Tell me a joke", SearchResult(chunks=CHUNKS, no_context=True), llm)
    assert llm.calls == []
    assert result.reply == NO_CONTEXT_REPLY
    assert result.sources == []
    assert result.steps[1]["data"]["provider"] is None


def test_llm_failure_gives_a_degraded_reply():
    llm = FakeLLM(error=RuntimeError("both providers down"))
    result = answer("What do you offer?", SearchResult(chunks=CHUNKS), llm)
    assert (result.reply, result.sources, result.degraded) == (LLM_DOWN_REPLY, [], True)


def test_keyword_fallback_search_is_degraded():
    result = answer("Q?", SearchResult(chunks=CHUNKS, mode="keyword", degraded=True), FakeLLM("A."))
    assert result.degraded is True
    assert result.reply == "A."


def test_full_width_citation_brackets_are_normalized():
    llm = FakeLLM("Steps: inquiry → handover【KB Section 8: Product & Service Catalog】.", provider="groq")
    result = answer("Process?", SearchResult(chunks=CHUNKS), llm)
    assert result.reply == "Steps: inquiry → handover[KB Section 8: Product & Service Catalog]."
    assert [s["section"] for s in result.sources] == ["8"]

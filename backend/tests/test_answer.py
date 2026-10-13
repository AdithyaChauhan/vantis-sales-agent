"""Prompts, context format, citations and sources (the answer helpers the agent reuses)."""
import re

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
    is_claim,
    soften_promises,
    strip_tool_citations,
    strip_unsourced_citations,
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


class ReasonLLM(FakeLLM):
    """A fake reason model that, as the reason prompt asks, ends its sentence with the prompt's "not … yet"
    part for a lead that is not HOT (T12a). For a HOT lead it returns the reply as it is."""

    def invoke(self, messages):
        message = super().invoke(messages)
        gap = re.search(r"Missing for the next status: (not \w+ yet: [^\n]*)", str(messages[0].content))
        if gap:
            message = message.model_copy(update={"content": f"{self.reply.rstrip('.')}, {gap.group(1)}."})
        return message


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
    # T13b: the package rule sits right after the context on a bill turn; on other turns its line goes
    assert "</context>\n\nThe lead progress" in rag_message("Hi", []).content
    with_rule = rag_message("Hi", [], package_guidance="<package_guidance>x</package_guidance>").content
    assert "</context>\n\n<package_guidance>x</package_guidance>\n\nThe lead progress" in with_rule


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


def test_a_reply_that_cites_nothing_has_no_sources():
    # T12a: a question or a confirmation used to get every context section as its sources ("KB 10" on a question)
    assert pick_sources("Could you share your name and phone number?", CHUNKS) == []
    assert pick_sources("Only [KB Section 99: Not In Context].", CHUNKS) == []


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


# ---------- T12a: KB tags only on facts from this turn's sources ----------

def test_a_tag_whose_section_is_not_among_this_turns_chunks_is_removed():
    reply = "HomePro 5 is 5 kW [KB Section 8: Product & Service Catalog]. Warranty varies [KB Section 11: Warranty]."
    assert strip_unsourced_citations(reply, [CATALOG]) == (
        "HomePro 5 is 5 kW [KB Section 8: Product & Service Catalog]. Warranty varies.", 1)


def test_a_turn_without_chunks_keeps_no_tag():
    reply = "Thanks [KB Section 10: Locations]! Your number looks short.\nCould you check it? [KB Section 1: Overview]"
    assert strip_unsourced_citations(reply, []) == (
        "Thanks! Your number looks short.\nCould you check it?", 2)


def test_tags_on_questions_and_claims_are_removed_even_for_a_found_section():
    reply = ("HomePro 5 is 5 kW [KB Section 8: Product & Service Catalog]. "
             "I've saved your details [KB Section 8: Product & Service Catalog]. "
             "Would you like a quote? [KB Section 15: Frequently Asked Questions]")
    assert strip_unsourced_citations(reply, [CATALOG, FAQ]) == (
        "HomePro 5 is 5 kW [KB Section 8: Product & Service Catalog]. I've saved your details. "
        "Would you like a quote?", 2)


def test_a_tag_on_a_thank_you_is_removed():
    # T12a replay: "Thank you for your interest in installing a solar system for your organization [KB Section 2]"
    reply = ("Thank you for your interest in solar for your organisation [KB Section 8: Product & Service Catalog]. "
             "HomePro 5 is 5 kW [KB Section 8: Product & Service Catalog].")
    assert strip_unsourced_citations(reply, [CATALOG]) == (
        "Thank you for your interest in solar for your organisation. "
        "HomePro 5 is 5 kW [KB Section 8: Product & Service Catalog].", 1)


def test_supported_tags_and_lists_are_left_exactly_as_they_are():
    reply = "Packages:\n1. HomeStart 3\n2. HomePro 5 [KB Section 8: Product & Service Catalog]\n\nAsk me more."
    assert strip_unsourced_citations(reply, [CATALOG]) == (reply, 0)


@pytest.mark.parametrize("sentence, claim", [
    ("Your details are saved.", True),
    ("I have passed your details to our sales team.", True),
    ("We've logged your callback request.", True),
    ("I've shared the steps above.", False),  # handing on counts only when a team or person got it
    ("Your details will be saved once you share a phone number.", False),
    ("Could you share your phone number?", False),
])
def test_save_claims(sentence, claim):
    assert is_claim(sentence) is claim


@pytest.mark.parametrize("reply, softened", [
    # T12a replay, on Groq: "Your details have been saved and our sales team will follow up with you shortly."
    ("Our sales team will follow up with you shortly.", "Our sales team can follow up with you shortly."),
    ("We'll call you tomorrow.", "We can call you tomorrow."),
    ("Someone will get back to you.", "Someone can get back to you."),
])
def test_a_promise_of_a_human_action_becomes_can(reply, softened):
    assert soften_promises(reply) == (softened, 1)


@pytest.mark.parametrize("reply", [
    "The team can follow up after a site assessment.",
    "The site assessment will confirm the final size.",  # not a person's action
    "Will you call us back?",
])
def test_other_wording_is_left_alone(reply):
    assert soften_promises(reply) == (reply, 0)


# ---------- T12b: "service areas", not the KB's "reference locations" ----------

def test_reference_location_words_become_service_areas_outside_tags():
    from app.rag.answer import plain_service_area_words

    reply = ("Delhi is a reference service area. Reference locations include Kota "
             "[KB Section 10: Reference Service Locations].")
    text, changed = plain_service_area_words(reply)
    assert changed == 2
    assert text == "Delhi is a service area. Service areas include Kota [KB Section 10: Reference Service Locations]."
    assert plain_service_area_words("Vantis covers Kota [KB Section 10: Reference Service Locations].")[1] == 0


def test_every_model_reads_the_checklist_last_and_offers_stay_answerable():
    # T12b: Groq (the fallback) invented durations and asked two questions; the rules now close the message
    from app.rag.answer import rag_message

    text = rag_message("How long does it take?", []).content
    assert text.index("How long does it take?") < text.index("Before you reply, check:")
    assert text.rstrip().endswith("At most one question.")
    system = load_prompt("system_prompt")
    assert "what each part covers" not in system  # an offer the KB can't answer (T12a run 3)
    assert "never \"reference locations\"" in system

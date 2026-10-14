"""The safety net (T13d): the reply when no model answered, built from this turn's KB chunks, the code's city checks,
what was saved and fixed texts. Synthetic chunks in the KB's shapes; no network, no database."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import welcome
from app.agent.fallback import candidates, fallback_reply, is_instruction, kb_sentences, key_words
from app.rag.answer import LLM_DOWN_REPLY
from app.rag.retriever import RetrievedChunk, SearchResult
from tests.agent_fakes import AREAS_SUMMARY, FakeSearch, Scripted, make_graph, run, understood


def chunk(chunk_id, section_no, title, chunk_type, content):
    return {"chunk_id": chunk_id, "section_no": section_no, "section_title": title, "page": 4,
            "chunk_type": chunk_type, "content": content}


WARRANTY_FAQ = chunk("s15-warranty", 15, "FAQ", "faq",
                     "Q: What warranty do solar panels have?\nA: Warranty coverage depends on the equipment and terms.")
SUBSIDY_FAQ = chunk("s15-subsidy", 15, "FAQ", "faq", "Q: Do you provide subsidy assistance?\n"
                    "A: The team can explain the considerations, but eligibility and amounts require verification.")
FINANCE_FAQ = chunk("s15-financing", 15, "FAQ", "faq", "Q: Do you provide financing?\n"
                    "A: Options may be available depending on the project. A representative can share more.")
PRICING = chunk("s05-pricing", 5, "Pricing", "prose",
                "Vantis keeps no single fixed public price for all installations. Cost depends on system size, "
                "equipment and location.\nNever fabricate an exact project price. Explain that a consultation is "
                "required to determine project-specific cost.")
PROCESS = chunk("s06-process", 6, "Installation", "prose",
                "General process: Inquiry → Assessment → Installation → Handover.\nA site assessment may evaluate "
                "rooftop availability, installation area and property conditions.\n"
                "Do not guarantee an installation date unless it is explicitly provided in the knowledge base.")
SUBSIDY_PROSE = chunk("s13-subsidies", 13, "Subsidies", "prose",
                      "Government subsidy programs and amounts may change over time. Customers should be told that "
                      "eligibility requires verification.")
INVERTER_ROW = chunk("s11-inverter", 11, "Warranty Policy", "table_row",
                     "Component / Area: Inverter\nReference Policy: Warranty depends on the selected equipment.")
SUPPORT_ROW = chunk("s12-inverter-issue", 12, "Support Scenarios", "table_row",
                    "Scenario: Customer reports inverter issue\nAI Response Direction: Collect identity and contact.")
CATALOG_ROW = chunk("s08-plan-b", 8, "Catalog", "table_row",
                    "Offering: Plan B\nPrimary Customer: Residential\nReference Profile: 5 kW rooftop solution")
CATALOG_SUMMARY = chunk("s08-table-summary", 8, "Catalog", "table_summary",
                        "All rows (Offering; Primary Customer; Reference Profile):\n"
                        "- Plan A; Residential; 3 kW rooftop solution\n- Plan B; Residential; 5 kW rooftop solution\n"
                        "- Plan C; Small Commercial; 15 kW reference solution")


def state(message, search_query="q", **extra):
    return {"messages": [HumanMessage(message)], "search_query": search_query, "lead": {}, "area_checks": [],
            "new_details": [], **extra}


@pytest.mark.parametrize("sentence", [
    "Never fabricate an exact project price.",
    "Explain that a consultation is required to determine project-specific cost.",
    "Customers should be told that eligibility requires verification.",
    "For sizing questions, collect relevant information before making a recommendation.",
    "The AI must not promise financing approval or a specific EMI.",
    "Do not guarantee an installation date unless it is explicitly provided in the knowledge base.",
    "Locations not listed should be treated as unverified.",
    "AI Response Direction: Collect identity and contact.",
])
def test_sentences_written_for_the_assistant_are_never_quoted(sentence):
    assert is_instruction(sentence)


@pytest.mark.parametrize("sentence", [
    "Cost depends on system size, equipment and location.",
    "Warranty coverage depends on the specific equipment and manufacturer terms.",
    "Recommended system size depends on electricity consumption and monthly bill.",
    "Vantis provides end-to-end solar solutions including consultation and installation.",
])
def test_facts_are_quotable(sentence):
    assert not is_instruction(sentence)


def test_each_chunk_type_gives_its_own_kind_of_sentence():
    assert [t for t, _ in candidates(WARRANTY_FAQ, None)] == [
        "Warranty coverage depends on the equipment and terms."]  # the answer, matched with its question
    assert [t for t, _ in candidates(PRICING, None)] == [
        "Vantis keeps no single fixed public price for all installations.",
        "Cost depends on system size, equipment and location."]  # no instructions
    assert [t for t, _ in candidates(INVERTER_ROW, None)] == ["Inverter: Warranty depends on the selected equipment."]
    assert candidates(SUPPORT_ROW, None) == []  # only an instruction besides its label
    assert candidates(CATALOG_ROW, None) == []  # one package alone: only for a settled size (all-or-none)
    assert [t for t, _ in candidates(CATALOG_ROW, "5")] == ["Plan B: 5 kW rooftop solution."]
    assert [t for t, _ in candidates(CATALOG_SUMMARY, None)] == [
        "1. Plan A (Residential)\n2. Plan B (Residential)\n3. Plan C (Small Commercial)"]
    assert [t for t, _ in candidates(CATALOG_SUMMARY, "5")] == ["Plan B: 5 kW rooftop solution."]


def test_key_words_match_hinglish_and_plurals_for_scoring_only():
    assert {"subsidy", "government"} <= key_words("subsidy milti hai kya sarkar se")
    assert "financing" in key_words("instalment me pay kar sakte hain")
    assert {"panel", "warranty"} <= key_words("panel ki warranty kitne saal ki hoti hai")
    assert key_words("kya hai aap log") == set()


def test_a_matching_faq_answer_comes_first_and_a_better_match_follows_it():
    picked = kb_sentences([INVERTER_ROW, WARRANTY_FAQ], "panel warranty", None)
    assert picked[0][1]["chunk_id"] == "s15-warranty"
    process = kb_sentences([PROCESS, chunk("s15-time", 15, "FAQ", "faq", "Q: How long does installation take?\n"
                                           "A: Timing varies with the site.")], "installation process steps", None)
    assert [text[:16] for text, _ in process] == ["Timing varies wi", "General process:"]


def test_no_shared_word_quotes_nothing_and_says_plainly_it_isnt_covered():
    reply, update, sections = fallback_reply(state("Do you sell water heaters?"), [PRICING, PROCESS], None)
    assert reply == f"{welcome.NOT_COVERED_TEXT}\n\n{welcome.FALLBACK_OFFER}"
    assert (update, sections) == ({"fallback_offered": True}, [])


def test_script_a_questions_get_their_kb_sentences_with_sources():
    reply, _, sections = fallback_reply(state("subsidy milti hai kya sarkar se"), [SUBSIDY_PROSE, SUBSIDY_FAQ], None)
    assert reply.startswith("The team can explain the considerations, but eligibility and amounts require "
                            "verification. [KB Section 15: FAQ] Government subsidy programs and amounts may change "
                            "over time. [KB Section 13: Subsidies]")
    assert "should be told" not in reply and sections == [15, 13]
    reply, _, _ = fallback_reply(state("instalment me pay kar sakte hain"), [FINANCE_FAQ], None)
    assert reply.startswith("Options may be available depending on the project. A representative can share more. "
                            "[KB Section 15: FAQ]")
    reply, _, _ = fallback_reply(state("chhat pe shade hai, kya problem hogi"), [PROCESS], None)
    assert reply.startswith("A site assessment may evaluate rooftop availability")


def test_a_settled_size_names_its_package_even_without_a_search():
    """The assignment's "5 kW for my house": the package for that size, from the KB Section 8 summary."""
    reply, _, sections = fallback_reply(state("I want a 5 kW system for my house", search_query=None),
                                        [CATALOG_SUMMARY], None)
    assert reply.startswith("Plan B: 5 kW rooftop solution. [KB Section 8: Catalog]") and sections == [8]


def test_city_lines_come_from_the_codes_check_with_the_kb_section_10_tag():
    checks = [{"city": "Noida", "status": "listed", "matched_city": "Noida"},
              {"city": "Pune", "status": "unverified", "matched_city": None}]
    reply, _, _ = fallback_reply(state("aap log noida me kaam karte ho, aur Pune?", area_checks=checks),
                                 [AREAS_SUMMARY, PRICING], None)
    first, second = reply.split("\n\n")[:2]
    assert first == "Vantis currently covers Noida. [KB Section 10: Locations]"
    assert second.startswith("I can't confirm service in Pune yet") and second.endswith("[KB Section 10: Locations]")


def test_a_city_the_message_doesnt_name_gets_no_line():
    checks = [{"city": "Lucknow", "status": "listed", "matched_city": "Lucknow"}]  # carried over by the extractor
    reply, _, _ = fallback_reply(state("Rahul, 9876543210", search_query=None, area_checks=checks,
                                       new_details=["name", "phone"]), [AREAS_SUMMARY], None)
    assert "Lucknow" not in reply and reply.startswith(welcome.DETAILS_THANKS_TEXT)


def test_what_was_saved_closes_the_reply_with_its_own_question():
    reply, update, _ = fallback_reply(state("Rahul, 9876543210", search_query=None, new_details=["phone"]), [],
                                      welcome.SAVED_FALLBACK_TEXT)
    assert (reply, update) == (welcome.SAVED_FALLBACK_TEXT, {})


def test_the_next_missing_detail_is_the_question_once_they_showed_interest():
    lead = {"customer_type": "residential"}
    reply, _, _ = fallback_reply(state("I want solar for my home", search_query=None, lead=lead,
                                       new_details=["customer_type"], intents=["buying_interest"]), [], None)
    assert reply == f"{welcome.DETAILS_THANKS_TEXT}\n\n{welcome.NEXT_QUESTION['city']}"


def test_the_offer_comes_once_per_chat_and_never_after_a_skip_or_a_no():
    plain = state("What is the warranty?")
    assert fallback_reply(plain, [WARRANTY_FAQ], None)[0].endswith(welcome.FALLBACK_OFFER)
    for after in ({"fallback_offered": True}, {"skipped_asks": 1}, {"ever_declined": True}):
        reply, update, _ = fallback_reply({**plain, **after}, [WARRANTY_FAQ], None)
        assert reply.endswith(welcome.ANYTHING_ELSE_ASK) and update == {}


def test_a_reply_is_never_only_a_question():
    for s in (state("ok", search_query=None), state("hmm", search_query=None, contact_request=True)):
        reply, _, _ = fallback_reply(s, [], None)
        assert "\n\n" in reply and not reply.split("\n\n")[0].endswith("?")


# ---------- through the graph: every model fails ----------

def test_every_model_failing_gets_the_kb_answer_with_its_chip_never_try_again():
    search = FakeSearch(SearchResult(chunks=[RetrievedChunk(**WARRANTY_FAQ), RetrievedChunk(**INVERTER_ROW)],
                                     mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question"], "panel warranty")), Scripted(RuntimeError("all down")),
                          search=search)
    result = run(graph, "panel ki warranty kitne saal ki hoti hai")
    assert result["reply"] != LLM_DOWN_REPLY
    assert result["reply"].startswith("Warranty coverage depends on the equipment and terms. [KB Section 15: FAQ]")
    assert [s["section"] for s in result["sources"]] == ["15"]
    assert (result["answered_by"], result["degraded"]) == ("knowledge_base", True)
    assert result["steps"][-1] == {"node": "fallback", "detail": "no model answered: reply from KB Section 15",
                                   "data": {"llm": False, "sections": [15]}}


def test_understand_and_the_answer_failing_still_search_the_raw_hinglish_message():
    """Item 2 (Adithya): with understand down, the KB search runs on the raw message even without a "?"."""
    search = FakeSearch(SearchResult(chunks=[RetrievedChunk(**SUBSIDY_FAQ)], mode="hybrid"))
    graph, _ = make_graph(Scripted(RuntimeError("understand down")), Scripted(RuntimeError("all down")),
                          search=search)
    result = run(graph, "subsidy milti hai kya sarkar se")
    assert search.queries == ["subsidy milti hai kya sarkar se"]
    assert "[KB Section 15: FAQ]" in result["reply"] and result["answered_by"] == "knowledge_base"


def test_the_offer_is_remembered_across_turns():
    search = FakeSearch(SearchResult(chunks=[RetrievedChunk(**WARRANTY_FAQ)], mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question"], "panel warranty")), Scripted(RuntimeError("all down")),
                          search=search)
    first = run(graph, "What warranty do the panels have?", thread="offer")
    second = run(graph, "And the warranty on panels again?", thread="offer")
    assert first["reply"].endswith(welcome.FALLBACK_OFFER)
    assert second["reply"].endswith(welcome.ANYTHING_ELSE_ASK)


def test_a_fixed_reply_turn_is_answered_by_fixed_text():
    graph, _ = make_graph(Scripted(understood(["greeting"])), Scripted(AIMessage("unused")))
    assert run(graph, "hi")["answered_by"] == "fixed"



def test_the_safety_net_lists_the_offerings_for_a_question_about_the_offer():
    # T13e: the understand step's asks_offerings flag (its word-list backup when that step failed) says so
    asks = {**state("What solar systems do you provide?"), "asks_offerings": True}
    reply, _, sections = fallback_reply(asks, [PRICING, CATALOG_SUMMARY], None)
    assert reply.startswith("1. Plan A (Residential)\n2. Plan B (Residential)\n3. Plan C (Small Commercial) "
                            "[KB Section 8: Catalog]") and sections[0] == 8

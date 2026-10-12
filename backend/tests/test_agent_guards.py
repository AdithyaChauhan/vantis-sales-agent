"""T12b guards and gates in the agent: own-details questions (D1), the catalog for a stated size (D4), the number
guard and the one-question guard in finalize. Fake models and search only."""
import pytest

from app.agent.leads import keep_one_question, own_details_question
from app.agent.welcome import NUMBERS_CONFIRM_TEXT
from app.rag.retriever import RetrievedChunk, SearchResult
from tests.agent_fakes import FAQ, FakeSearch, Scripted, answer, make_graph, run, understood

PRICING = RetrievedChunk("s05-pricing", 5, "Pricing", 2, "prose", "There is no single fixed price.")
TIMING = RetrievedChunk("s15-timing", 15, "FAQ", 5, "faq", "Q: How long does installation take?\nA: Timing varies.")
PLAN_A = RetrievedChunk("s08-plan-a", 8, "Catalog", 3, "table_row", "Offering: Plan A, 3 kW")


# ---------- D1: "what is needed" only for the customer's own details ----------

@pytest.mark.parametrize("message, own", [
    ("what else do you need from me?", True),
    ("is there any other info u need that can help apart from timing?", True),
    ("are you sure? rooftop availability info is not needed?", True),
    ("do you need my email too?", True),
    ("What else do I need for installation?", False),
    ("Do you offer anything else besides panels?", False),
    ("what documents are required?", False),
    ("What do I need for installation?", False),
])
def test_own_details_question(message, own):
    assert own_details_question(message) is own


def test_a_kb_question_labelled_asks_what_is_needed_still_searches():
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(["asks_what_is_needed"])), Scripted(answer("You need a site visit.")),
                          search=search)
    state = run(graph, "What else do I need for installation?")
    assert search.queries == ["What else do I need for installation?"]
    assert "question" in state["intents"] and "asks_what_is_needed" not in state["intents"]
    assert state["steps"][0]["data"]["query_note"] == "not about their details: searches the KB"


def test_what_else_do_you_need_from_me_does_not_search():
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(["asks_what_is_needed"])), Scripted(answer("Your city, please?")),
                          search=search)
    run(graph, "what else do you need from me?")
    assert search.queries == []


def test_a_question_without_a_query_searches_with_the_message():
    # T12a eval: q13 once came back with no search query at all
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(["question", "existing_customer_issue"])),
                          Scripted(answer("Sorry to hear that.")), search=search)
    state = run(graph, "My inverter has stopped working. What should I do?")
    assert search.queries == ["My inverter has stopped working. What should I do?"]
    assert state["steps"][0]["data"]["query_note"] == "no query from the model: the message is the query"


@pytest.mark.parametrize("intents", [["off_topic", "question"], ["reply_to_assistant", "question"],
                                     ["contact_request", "question"]])
def test_no_raw_message_search_for_off_topic_short_replies_or_contact_requests(intents):
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(intents)), Scripted(answer("Okay.")), search=search)
    run(graph, "yes please")
    assert search.queries == []


# ---------- D4: a stated size brings the catalog, so the package can be named ----------

def test_a_size_in_the_message_adds_the_catalog_summary():
    agent = Scripted(answer("Pricing depends on the project [KB Section 5: Pricing]."))
    search = FakeSearch(SearchResult(chunks=[PRICING], mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question"], "Pricing cost")), agent, search=search)
    state = run(graph, "How much does a 5kW system cost?")
    context = agent.calls[0][-1].content
    assert "Plan B 5 kW" in context  # the KB Section 8 summary (tests/agent_fakes.py)
    assert [c["chunk_id"] for c in state["chunks"]] == ["s05-pricing", "s08-table-summary"]


def test_no_size_no_catalog():
    search = FakeSearch(SearchResult(chunks=[PRICING], mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question"], "Pricing cost")), Scripted(answer("It depends.")),
                          search=search)
    assert [c["chunk_id"] for c in run(graph, "How much does solar cost?")["chunks"]] == ["s05-pricing"]


# ---------- the number guard and the one-question guard in finalize ----------

def test_finalize_removes_an_invented_duration_and_says_the_team_confirms():
    agent = Scripted(answer("Timing varies with the site [KB Section 15: FAQ]. Most installs take a few weeks."))
    search = FakeSearch(SearchResult(chunks=[TIMING], mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question"], "installation timing")), agent, search=search)
    state = run(graph, "How long does installation take?")
    assert state["reply"] == f"Timing varies with the site [KB Section 15: FAQ]. {NUMBERS_CONFIRM_TEXT}"
    assert any(s["node"] == "guard" and "number" in s["detail"] for s in state["steps"])


def test_a_number_from_an_earlier_reply_may_be_repeated_without_a_search():
    search = FakeSearch(SearchResult(chunks=[PLAN_A], mode="hybrid"))
    agent = Scripted(answer("Plan A is a 3 kW reference option [KB Section 8: Catalog]."),
                     answer("Plan A, the 3 kW option, is a starting point; a site assessment sets the size."))
    understand = Scripted(understood(["question"], "catalog"), understood(["reply_to_assistant"]))
    graph, _ = make_graph(understand, agent, search=search)
    run(graph, "What do you offer?")
    state = run(graph, "ok, and that one?")
    assert state["reply"] == "Plan A, the 3 kW option, is a starting point; a site assessment sets the size."


def test_finalize_keeps_one_question_the_closing_offer():
    agent = Scripted(answer("Plan A is a reference option [KB Section 8: Catalog]. Is that helpful? "
                            "Would you like a callback from the team?"))
    graph, _ = make_graph(Scripted(understood(["question"], "catalog")), agent)
    state = run(graph, "What do you offer?")
    assert state["reply"] == ("Plan A is a reference option [KB Section 8: Catalog]. "
                              "Would you like a callback from the team?")
    assert any(s["detail"] == "kept one question" for s in state["steps"])


def test_finalize_keeps_the_reply_when_it_asks_one_question():
    graph, _ = make_graph(Scripted(understood(["question"], "faq")),
                          Scripted(answer("Yes [KB Section 15: FAQ]. More?")),
                          search=FakeSearch(SearchResult(chunks=[FAQ], mode="hybrid")))
    assert run(graph, "Can I get a quote?")["reply"] == "Yes [KB Section 15: FAQ]. More?"


@pytest.mark.parametrize("state, kept", [
    # interest shown, no lead yet: the next missing item is the city
    ({"intents": ["buying_interest"], "lead": {}}, "Which city is your home in?"),
    # information only: the last question (the closing offer)
    ({"intents": ["question"], "lead": {}}, "Would you like a callback?"),
    # lead saved, timeline missing: the timeline question
    ({"intents": ["gives_details"], "lead_id": 7, "new_details": ["email"],
      "lead": {"name": "Asha", "phone": "+919876543210", "email": "a@example.com", "city": "Lucknow",
               "monthly_bill_inr": 3000, "ownership": "own", "rooftop_available": True}},
     "When would you like it installed?"),
])
def test_keep_one_question_picks_the_next_detail_else_the_last(state, kept):
    first = "When would you like it installed?" if state.get("lead_id") else "Which city is your home in?"
    reply = f"Thanks. {first} Is that helpful? Would you like a callback?"
    text, changed = keep_one_question(reply, {"messages": [], **state})
    assert changed and text == f"Thanks. {kept}"


def test_keep_one_question_drops_the_tags_of_a_dropped_question_only():
    reply = "Plan A fits [KB Section 8: Catalog]. Want the list? [KB Section 8: Catalog]\nShall I check your city?"
    text, _ = keep_one_question(reply, {"intents": ["question"], "lead": {}, "messages": []})
    assert text == "Plan A fits [KB Section 8: Catalog].\nShall I check your city?"


def test_the_support_step_shows_the_request_id():
    from app.agent.tools import _lead_tool_detail

    assert _lead_tool_detail('{"saved": true, "support_request_id": 4}') == "saved: request 4"
    assert _lead_tool_detail('{"saved": true, "fields_saved": ["city"]}') == "saved: city"


# ---------- every city named is checked in code before the model (T12b, the slow-Google drill) ----------

def test_every_city_named_is_checked_before_the_model_with_its_step_and_source():
    agent = Scripted(answer("Vantis currently covers Lucknow, and the team can check Pune "
                            "[KB Section 10: Locations]."))
    graph, _ = make_graph(Scripted(understood(["question"], "service areas", cities=["Lucknow", "Pune"],
                                              asked_city="Lucknow")), agent)
    state = run(graph, "Do you work in Lucknow or Pune?")
    checks = [s for s in state["steps"] if s["node"] == "area_check"]
    assert [s["detail"] for s in checks] == ["Lucknow: listed", "Pune: unverified"]
    assert len(agent.calls) == 1  # no model round for a tool call
    message = agent.calls[0][-1].content
    assert '"city": "Lucknow", "status": "listed"' in message and '"city": "Pune", "status": "unverified"' in message
    assert state["chunks"][0]["chunk_id"] == "s10-table-summary"  # the city answer's source comes first
    assert [s["section"] for s in state["sources"]] == ["10"]


def test_a_stated_city_is_checked_on_a_details_turn_without_a_search():
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, city="Kanpur")),
                          Scripted(answer("Thanks! What is your monthly bill?")), search=search)
    state = run(graph, "Kanpur")
    assert search.queries == []
    assert [s["detail"] for s in state["steps"] if s["node"] == "area_check"] == ["Kanpur: listed"]


def test_no_city_check_for_an_off_topic_message_and_none_without_a_city():
    graph, _ = make_graph(Scripted(understood(["off_topic"], cities=["Delhi"])), Scripted(answer("unused")))
    assert not [s for s in run(graph, "weather in Delhi?")["steps"] if s["node"] == "area_check"]
    graph, _ = make_graph(Scripted(understood(["question"], "catalog")), Scripted(answer("Plan A.")))
    assert run(graph, "What do you offer?")["area_checks"] == []


def test_the_message_says_no_city_was_named_when_none_was():
    agent = Scripted(answer("Plan A."))
    graph, _ = make_graph(Scripted(understood(["question"], "catalog")), agent)
    run(graph, "What do you offer?")
    assert "(no city named)" in agent.calls[0][-1].content


def test_a_city_check_that_cannot_read_the_list_leaves_the_tool_to_the_model():
    from app.agent.graph import check_named_cities

    def down():
        raise RuntimeError("service_areas table is empty")

    assert check_named_cities(["Delhi"], down) == []


# ---------- a model failure after a tool wrote something (T12b) ----------

def _tool_result(name, artifact):
    from langchain_core.messages import ToolMessage

    return ToolMessage(content="{}", tool_call_id=f"c-{name}", name=name, artifact=artifact)


def test_tool_outcome_reply_says_what_was_written():
    from app.agent import welcome
    from app.agent.graph import tool_outcome_reply

    saved = _tool_result("create_lead", {"kind": "lead", "lead_id": 7, "saved_fields": {}, "created": True})
    callback = _tool_result("schedule_callback", {"kind": "callback", "status": "requested", "time": "4pm",
                                                  "updated": False})
    support = _tool_result("create_support_request", {"kind": "support", "id": 12})
    city = _tool_result("check_service_area", [{"chunk_id": "s10-table-summary"}])
    assert tool_outcome_reply([saved]) == welcome.SAVED_FALLBACK_TEXT
    assert tool_outcome_reply([saved, callback]) == (
        "Your details are saved, and the Vantis sales team can follow up with you. "
        "I've logged a callback request for 4pm; the team will confirm it.")
    assert tool_outcome_reply([support]) == (
        "I've saved your support request (id 12); the Vantis team can follow up on it.")
    assert tool_outcome_reply([city]) is None and tool_outcome_reply([]) is None


def test_a_model_failure_after_only_a_lookup_is_still_try_again():
    from app.rag.answer import LLM_DOWN_REPLY
    from tests.agent_fakes import tool_call

    agent = Scripted(tool_call("check_service_area", {"city": "Pune"}), RuntimeError("down"))
    graph, _ = make_graph(Scripted(understood(["question"], "areas")), agent)
    assert run(graph, "Do you install there?")["reply"] == LLM_DOWN_REPLY


def test_the_number_guard_step_quotes_each_removed_sentence():
    agent = Scripted(answer("Timing varies with the site [KB Section 15: FAQ]. Most installs take a few weeks."))
    search = FakeSearch(SearchResult(chunks=[TIMING], mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question"], "installation timing")), agent, search=search)
    state = run(graph, "How long does installation take?")
    [step] = [s for s in state["steps"] if s["node"] == "guard" and "number" in s["detail"]]
    assert step["data"] == {"removed": ["Most installs take a few weeks."], "numbers": ["a few weeks"]}


# ---------- T13a: the package guidance (KB Section 9) on a turn that gives the monthly bill ----------

def test_a_bill_brings_the_package_guidance_so_the_suggestion_keeps_its_number_and_tag():
    # C7 and C7b turn 2: "That monthly bill fits nicely with our 3 kW residential reference package." was removed
    # by the number guard, because a details-only turn has no KB search and so no source for "3 kW"
    reply = ("Plan A could be a starting point for that bill, though 3 kW is not assumed without a site check "
             "[KB Section 9: Guidance]. Could you share your name and phone number?")
    agent = Scripted(answer(reply))
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, monthly_bill_inr="4000")), agent, search=search)
    state = run(graph, "My bill is about 4000 a month")
    assert search.queries == []  # still no KB search on a details-only turn
    assert "Moderate residential requirement" in agent.calls[0][-1].content
    assert [c["chunk_id"] for c in state["chunks"]] == ["s09-moderate", "s09-higher"]
    assert state["reply"] == reply  # the number and the KB tag both stay
    assert not [s for s in state["steps"] if s["node"] == "guard"]
    assert [s["section"] for s in state["sources"]] == ["9"]


def test_no_bill_this_turn_no_guidance():
    agent = Scripted(answer("Thanks."))
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, name="Arjun")), agent)
    assert run(graph, "I am Arjun")["chunks"] == []
    assert "Moderate residential requirement" not in agent.calls[0][-1].content


def test_a_bill_already_known_does_not_bring_it_again():
    understand = Scripted(understood(["gives_details"], None, monthly_bill_inr="4000"),
                          understood(["gives_details"], None, monthly_bill_inr="4000", name="Arjun"))
    graph, _ = make_graph(understand, Scripted(answer("Thanks.")))
    run(graph, "My bill is about 4000 a month")
    assert run(graph, "Arjun, and yes, 4000")["chunks"] == []


def test_a_bill_on_a_turn_with_a_search_adds_the_guidance_first():
    search = FakeSearch(SearchResult(chunks=[PRICING], mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question", "gives_details"], "Pricing", monthly_bill_inr="4000")),
                          Scripted(answer("It depends.")), search=search)
    state = run(graph, "My bill is 4000, how much would it cost?")
    assert [c["chunk_id"] for c in state["chunks"]] == ["s09-moderate", "s09-higher", "s05-pricing"]

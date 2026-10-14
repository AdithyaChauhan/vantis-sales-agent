"""T12b guards and gates in the agent: own-details questions (D1), the catalog for a stated size (D4), the number
guard and the one-question guard in finalize; T13a/T13b: the package guidance and rule on a bill turn. Fake models
and search only."""
import pytest
from langchain_core.messages import AIMessage

from app.agent.leads import keep_one_question, package_case, passing_place_kinds, place_type
from app.agent.welcome import NUMBERS_CONFIRM_TEXT
from app.core.llm import LLMError
from app.core.prompts import load_prompt
from app.rag.numbers import remove_unsupported_numbers
from app.rag.retriever import RetrievedChunk, SearchResult
from tests.agent_fakes import FAQ, FakeSearch, Scripted, answer, make_graph, run, understood

PRICING = RetrievedChunk("s05-pricing", 5, "Pricing", 2, "prose", "There is no single fixed price.")
TIMING = RetrievedChunk("s15-timing", 15, "FAQ", 5, "faq", "Q: How long does installation take?\nA: Timing varies.")
PLAN_A = RetrievedChunk("s08-plan-a", 8, "Catalog", 3, "table_row", "Offering: Plan A, 3 kW")


# ---------- D1: "what is needed" only for the customer's own details (T13e: the understand step decides) ----------

def test_a_kb_question_the_understand_step_reads_as_a_question_searches():
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(["question"], "Installation & Site Assessment requirements")),
                          Scripted(answer("You need a site visit.")), search=search)
    run(graph, "What else do I need for installation?")
    assert search.queries == ["Installation & Site Assessment requirements"]


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

@pytest.mark.parametrize("understanding", [
    understood(["question"], "Pricing cost Plan B 5 kW"),  # the understand step's query names an offering
    understood(["question", "buying_interest"], "Pricing cost", system_size_kw="5 kW"),  # or the size is new
])
def test_a_size_in_the_message_adds_the_catalog_summary(understanding):
    # T13e: read from the understand step's fields (its query, a new size), not from the message's words
    agent = Scripted(answer("Pricing depends on the project [KB Section 5: Pricing]."))
    search = FakeSearch(SearchResult(chunks=[PRICING], mode="hybrid"))
    graph, _ = make_graph(Scripted(understanding), agent, search=search)
    state = run(graph, "How much does a 5kW system cost?")
    context = agent.calls[0][-1].content
    assert "Plan B; Residential; 5 kW" in context  # the KB Section 8 summary (tests/agent_fakes.py)
    # T13d: the understand step adds it (also without a search), so it comes before the search's chunks
    assert [c["chunk_id"] for c in state["chunks"]] == ["s08-table-summary", "s05-pricing"]


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


def test_a_reply_whose_every_sentence_has_an_unsourced_number_is_never_blank():
    # T13c: a count or a size ("7 kW", "14 panels") is removed without the fixed line, so this reply came out empty
    reply = "A 7 kW system would suit you [KB Section 8: Catalog]. It needs 14 panels."
    assert remove_unsupported_numbers(reply, [], NUMBERS_CONFIRM_TEXT)[0] == ""  # the number guard alone
    graph, _ = make_graph(Scripted(understood(["question"], "system size")), Scripted(answer(reply)))
    state = run(graph, "What size do I need?")
    assert state["reply"] == NUMBERS_CONFIRM_TEXT and state["sources"] == []
    guards = [s for s in state["steps"] if s["node"] == "guard"]
    assert guards[0]["data"]["removed"] == ["A 7 kW system would suit you.", "It needs 14 panels."]
    assert guards[-1]["detail"] == "nothing was left after the guards; the fixed line replaced it"


def test_a_number_from_an_earlier_reply_may_be_repeated_without_a_search():
    search = FakeSearch(SearchResult(chunks=[PLAN_A], mode="hybrid"))
    # T13c: both fake packages named, since a reply with one of them and no settled size loses it (all or none)
    agent = Scripted(answer("Plan A is a 3 kW reference option, and Plan B is the other [KB Section 8: Catalog]."),
                     answer("The 3 kW option is a starting point; a site assessment sets the size."))
    understand = Scripted(understood(["question"], "catalog"), understood(["reply_to_assistant"]))
    graph, _ = make_graph(understand, agent, search=search)
    run(graph, "What do you offer?")
    state = run(graph, "ok, and that one?")
    assert state["reply"] == "The 3 kW option is a starting point; a site assessment sets the size."


def test_finalize_keeps_one_question_the_closing_offer():
    agent = Scripted(answer("Plan A and Plan B are the reference options [KB Section 8: Catalog]. Is that helpful? "
                            "Would you like a callback from the team?"))
    graph, _ = make_graph(Scripted(understood(["question"], "catalog")), agent)
    state = run(graph, "What do you offer?")
    assert state["reply"] == ("Plan A and Plan B are the reference options [KB Section 8: Catalog]. "
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


def test_a_model_failure_after_only_a_lookup_gets_the_safety_net_with_that_city():
    """T13d: never "I'm having trouble": the city the model checked with the tool is said in fixed words, with its
    source (the lookup wrote nothing, so the turn is not a tool outcome)."""
    from app.rag.answer import LLM_DOWN_REPLY
    from tests.agent_fakes import tool_call

    agent = Scripted(tool_call("check_service_area", {"city": "Pune"}), RuntimeError("down"))
    graph, _ = make_graph(Scripted(understood(["question"], "areas")), agent)
    state = run(graph, "Do you install there?")
    assert state["reply"] != LLM_DOWN_REPLY
    assert state["reply"].startswith("I can't confirm service in Pune yet") and "[KB Section 10: Locations]" in state[
        "reply"]
    assert (state["answered_by"], state["degraded"]) == ("knowledge_base", True)
    assert [s["section"] for s in state["sources"]] == ["10"]


def test_the_number_guard_step_quotes_each_removed_sentence():
    agent = Scripted(answer("Timing varies with the site [KB Section 15: FAQ]. Most installs take a few weeks."))
    search = FakeSearch(SearchResult(chunks=[TIMING], mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question"], "installation timing")), agent, search=search)
    state = run(graph, "How long does installation take?")
    [step] = [s for s in state["steps"] if s["node"] == "guard" and "number" in s["detail"]]
    assert step["data"] == {"removed": ["Most installs take a few weeks."], "numbers": ["a few weeks"]}


# ---------- T13a: the package guidance (KB Section 9) on a turn that gives the monthly bill ----------
# T13b: a bill alone never picks a package, so the rows come with the package rule, once the property type is known

HOME_BILL = dict(monthly_bill_inr="4000", property_type="house")
RULE = "<package_guidance>"  # the block from prompts/package_guidance_prompt.md


def agent_input(agent: Scripted, call: int = 0) -> str:
    """The customer's message as the agent got it (rag_prompt.md filled in)."""
    return str(agent.calls[call][-1].content)


def test_a_home_bill_lists_every_home_package_with_the_section_9_tag():
    # C7 turn 2: the same bill got one package picked by guess (HomeStart 3 locally, HomePro 5 on Render); the reply
    # now names every home package and leaves the choice to the site assessment
    reply = ("For a home, Vantis has two reference packages:\n1. Plan A: moderate residential requirement\n"
             "2. Plan B: higher residential requirement [KB Section 9: Guidance]\n"
             "A site assessment decides which one fits. Could you share your name and phone number?")
    agent = Scripted(answer(reply))
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, **HOME_BILL)), agent, search=search)
    state = run(graph, "My bill is about 4000 a month, it is for my home")
    assert search.queries == []  # still no KB search on a details-only turn
    assert [c["chunk_id"] for c in state["chunks"]] == ["s09-moderate", "s09-higher"]
    message = agent_input(agent)
    assert "Moderate residential requirement" in message and "Property type: home." in message
    assert message.index("</context>") < message.index(RULE) < message.index("<lead_progress>")
    assert state["reply"] == reply  # the list and its KB tag stay
    assert not [s for s in state["steps"] if s["node"] == "guard"]
    assert [s["section"] for s in state["sources"]] == ["9"]


def test_a_bill_without_the_property_type_asks_home_or_business_first():
    agent = Scripted(answer("Thanks. Is the solar for a home or a business?"))
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, monthly_bill_inr="4000")), agent)
    state = run(graph, "My bill is about 4000 a month")
    assert state["chunks"] == []  # no package rows while the property type is unknown
    assert "Property type: unknown." in agent_input(agent)


def test_the_property_type_after_the_bill_brings_the_packages():
    understand = Scripted(understood(["gives_details"], None, monthly_bill_inr="4000"),
                          understood(["gives_details"], None, **HOME_BILL))
    agent = Scripted(answer("Is it for a home or a business?"), answer("Thanks."))
    graph, _ = make_graph(understand, agent)
    run(graph, "My bill is about 4000 a month")
    state = run(graph, "home")
    assert [c["chunk_id"] for c in state["chunks"]] == ["s09-moderate", "s09-higher"]
    assert "Property type: home." in agent_input(agent, 1)


def test_a_business_bill_gets_the_business_rule():
    agent = Scripted(answer("Thanks."))
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, monthly_bill_inr="9000", property_type="shop")),
                          agent)
    state = run(graph, "The bill is 9000, it is for my shop")
    assert "Property type: business." in agent_input(agent)
    assert len(state["chunks"]) == 2


@pytest.mark.parametrize("intents, details", [
    (["gives_details"], dict(HOME_BILL, system_size_kw="5 kW")),  # a settled size: the stated-size rule names it
    (["contact_request", "gives_details"], HOME_BILL),  # name and phone come first
    (["existing_customer_issue"], dict(HOME_BILL, issue="inverter fault")),  # a support case (T13e: with its issue)
])
def test_no_package_rule_with_a_settled_size_a_contact_request_or_a_support_case(intents, details):
    agent = Scripted(answer("Thanks."))
    graph, _ = make_graph(Scripted(understood(intents, None, **details)), agent)
    state = run(graph, "My bill is 4000 for my home")
    # T13d: a size stated this turn brings the KB Section 8 summary (so the 5 kW package can be named), never the
    # Section 9 rows of the bill rule
    sized = "system_size_kw" in details and "existing_customer_issue" not in intents
    assert [c["chunk_id"] for c in state["chunks"]] == (["s08-table-summary"] if sized else [])
    assert RULE not in agent_input(agent) and "{package_guidance}" not in agent_input(agent)


def test_no_bill_this_turn_no_guidance():
    agent = Scripted(answer("Thanks."))
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, name="Arjun")), agent)
    assert run(graph, "I am Arjun")["chunks"] == []
    assert "Moderate residential requirement" not in agent_input(agent) and RULE not in agent_input(agent)


def test_a_bill_already_known_does_not_bring_it_again():
    understand = Scripted(understood(["gives_details"], None, **HOME_BILL),
                          understood(["gives_details"], None, **HOME_BILL, name="Arjun"))
    graph, _ = make_graph(understand, Scripted(answer("Thanks.")))
    assert len(run(graph, "My bill is about 4000 a month, it is for my home")["chunks"]) == 2
    assert run(graph, "Arjun, and yes, 4000")["chunks"] == []


def test_a_bill_on_a_turn_with_a_search_adds_the_guidance_first():
    search = FakeSearch(SearchResult(chunks=[PRICING], mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question", "gives_details"], "Pricing", **HOME_BILL)),
                          Scripted(answer("It depends.")), search=search)
    state = run(graph, "My bill is 4000 for my home, how much would it cost?")
    assert [c["chunk_id"] for c in state["chunks"]] == ["s09-moderate", "s09-higher", "s05-pricing"]


@pytest.mark.parametrize("state, case", [
    ({"lead": {"monthly_bill_inr": 4000, "customer_type": "residential"}, "new_details": ["monthly_bill_inr"]}, "home"),
    ({"lead": {"monthly_bill_inr": 9000, "customer_type": "commercial"}, "new_details": ["monthly_bill_inr"]},
     "business"),
    ({"lead": {"monthly_bill_inr": 4000}, "new_details": ["monthly_bill_inr"]}, "unknown"),
    ({"lead": {"monthly_bill_inr": 4000, "property_type": "other"}, "new_details": ["monthly_bill_inr"]}, "unknown"),
    ({"lead": {"monthly_bill_inr": 4000, "property_type": "house", "customer_type": "residential"},
      "new_details": ["customer_type", "property_type"]}, "home"),  # the type given after the bill
    ({"lead": {"monthly_bill_inr": 4000, "system_size_kw": 5.0}, "new_details": ["monthly_bill_inr"]}, None),
    ({"lead": {"monthly_bill_inr": 4000}, "new_details": ["monthly_bill_inr"], "support_case": True}, None),
    ({"lead": {"monthly_bill_inr": 4000}, "new_details": ["monthly_bill_inr"], "contact_request": True}, None),
    ({"lead": {"customer_type": "residential"}, "new_details": ["customer_type"]}, None),  # no bill yet
    ({"lead": {"monthly_bill_inr": 4000, "customer_type": "residential"}, "new_details": ["name"]}, None),
])
def test_package_case(state, case):
    assert package_case(state) == case


def test_a_bill_turn_without_the_property_type_keeps_the_home_or_business_question():
    # Without it, keep_one_question keeps the question for the next missing detail (the name) and drops this one
    state = {"lead": {"monthly_bill_inr": 4000, "city": "Lucknow"}, "new_details": ["monthly_bill_inr"],
             "intents": ["gives_details"]}
    reply = "Thanks. Is the solar for a home or a business? Could you share your name and phone number?"
    assert keep_one_question(reply, state) == ("Thanks. Is the solar for a home or a business?", True)
    known = {**state, "lead": {**state["lead"], "customer_type": "residential"}}
    assert keep_one_question(reply, known)[0] == "Thanks. Could you share your name and phone number?"


def test_the_prompts_carry_the_package_rule():
    system = load_prompt("system_prompt")
    assert "A monthly bill alone never picks a package" in system
    assert "which reference package fits their monthly bill" not in system  # the offer that invited the guess
    rule = load_prompt("package_guidance_prompt")
    assert rule.startswith(RULE) and "Property type: {property}." in rule
    assert "{package_guidance}" in load_prompt("rag_prompt")
    extraction = load_prompt("lead_extraction_prompt")
    assert "fills customer_type: residential for a home" in extraction
    assert "never from a passing mention" in extraction  # T13c
    assert "Reference packages, all or none" in system and "Never state a policy either way" in system  # T13c


# ---------- T13c: the kind of place only with confidence (place_type), and the type guard ----------

HOME_OR_BUSINESS = AIMessage("Thanks! Is the solar for a house or a small business?")


@pytest.mark.parametrize("message, previous, kind", [
    # (a) a short answer to the assistant's own home-or-business question (chat 1: "house", "home", "home")
    ("house", HOME_OR_BUSINESS, "residential"),
    ("home", HOME_OR_BUSINESS, "residential"),
    ("ghar", HOME_OR_BUSINESS, "residential"),
    ("a small shop in Lucknow", HOME_OR_BUSINESS, "commercial"),
    ("yes", HOME_OR_BUSINESS, None),
    ("home", None, None),  # a bare word nobody asked for
    # (b) clearly their own place
    ("actually the house is in Kanpur", None, "residential"),  # chat 1: the bot asked "house or small business?" next
    ("it's for my home", None, "residential"),
    ("My bill is about 4000 a month, it is for my home.", None, "residential"),
    ("my flat", None, "residential"),
    ("We live in a flat in Noida", None, "residential"),
    ("solar lagwana hai ghar par. kitna kharcha hoga", None, "residential"),  # chat 1, turn 1
    ("my bungalow in Pune", None, "residential"),
    ("we have a duplex", None, "residential"),
    ("it's a 2 BHK in Noida", None, "residential"),
    ("it's a shop", None, "commercial"),
    ("It is for our store", None, "commercial"),
    ("The solar is for my office", None, "commercial"),
    # not guessed: both kinds, neither, a plain question, a phrase that only looks like one
    ("my home office", None, None),
    ("My bill is 4000", None, None),
    ("Do you have a package for a business?", None, None),
    ("that's a business decision", None, None),
    ("main ghar par hoon", None, None),  # "I'm at home", nothing about solar
])
def test_place_type(message, previous, kind):
    assert place_type(message, previous) == ({"customer_type": kind} if kind else {})


NEVER = ["I'm at home right now", "my friend's house has solar", "I work from home",
         "Do you help with a home loan?"]


@pytest.mark.parametrize("message", NEVER)
def test_a_passing_mention_never_gives_the_type(message):
    # Adithya's "never" list (T13c), also as a short answer to the home-or-business question
    assert place_type(message, None) == {} and place_type(message, HOME_OR_BUSINESS) == {}
    assert passing_place_kinds(message) == {"residential"}


@pytest.mark.parametrize("message", NEVER)
def test_a_passing_mention_gives_no_type_when_the_understand_step_failed(message):
    # T13e: the understand step decides the kind of place (its prompt names passing mentions); the words are the backup
    graph, _ = make_graph(Scripted(LLMError("all down")), Scripted(answer("Thanks.")))
    state = run(graph, message)
    assert "customer_type" not in state["lead"]


def test_a_place_of_their_own_next_to_a_passing_mention_keeps_the_type():
    # Adithya (planning): "at home now, it's for my 2BHK" is a home of their own
    assert place_type("at home now, it's for my 2BHK", None) == {"customer_type": "residential"}
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, customer_type="residential")),
                          Scripted(answer("Thanks.")))
    state = run(graph, "at home now, it's for my 2BHK")
    assert state["lead"]["customer_type"] == "residential"


def test_a_passing_mention_never_drops_a_type_from_an_earlier_turn():
    understand = Scripted(understood(["gives_details"], None, property_type="shop"),
                          understood(["gives_details"], None, property_type="shop", monthly_bill_inr="9000"))
    graph, _ = make_graph(understand, Scripted(answer("Thanks.")))
    run(graph, "It is for my shop")
    state = run(graph, "The bill is 9000, I mostly work from home")
    assert state["lead"]["customer_type"] == "commercial"


def test_a_short_answer_counts_when_the_understand_step_failed():
    # Chat 1: "house" after "house or small business?" came right after a model failure, so no intent was known
    agent = Scripted(answer("Is the solar for a house or a small business?"), answer("Thanks."))
    understand = Scripted(understood(["gives_details"], None, monthly_bill_inr="6000"), LLMError("all down"))
    graph, _ = make_graph(understand, agent)
    run(graph, "my bill is 6k")
    state = run(graph, "house")
    assert state["steps"][0]["detail"] == "failed, raw message used"
    assert state["lead"]["customer_type"] == "residential" and state["steps"][0]["data"]["from_words"] == [
        "customer_type"]


def test_a_home_on_the_bill_turn_brings_the_home_packages():
    # C7 run 1: "it's for my home" stored no type on the bill turn, so the reply searched twice and cited KB Section 8.
    # T13e: the understand step gives the type (its place rules); the words are read only when that step failed
    agent = Scripted(answer("Thanks."))
    graph, _ = make_graph(Scripted(understood(["gives_details", "buying_interest"], None, monthly_bill_inr="4000",
                                              customer_type="residential")), agent)
    state = run(graph, "My bill is about 4000 a month, it's for my home.")
    assert state["lead"]["customer_type"] == "residential" and state["steps"][0]["data"]["from_words"] == []
    assert len(state["chunks"]) == 2 and "Property type: home." in agent_input(agent)
    agent = Scripted(answer("Thanks."))
    graph, _ = make_graph(Scripted(LLMError("all down")), agent)
    state = run(graph, "My bill is about 4000 a month, it's for my home.")
    assert state["lead"]["customer_type"] == "residential"
    assert state["steps"][0]["data"]["from_words"] == ["customer_type"]


def test_the_words_never_replace_a_type_from_the_extractor_or_an_earlier_turn():
    understand = Scripted(understood(["gives_details"], None, property_type="shop"),
                          understood(["gives_details"], None, monthly_bill_inr="9000"))
    graph, _ = make_graph(understand, Scripted(answer("Thanks.")))
    first = run(graph, "It is my shop, next to my home")
    assert first["lead"]["customer_type"] == "commercial" and first["steps"][0]["data"]["from_words"] == []
    second = run(graph, "The bill is 9000, I mostly work from home")
    assert second["lead"]["customer_type"] == "commercial" and second["steps"][0]["data"]["from_words"] == []

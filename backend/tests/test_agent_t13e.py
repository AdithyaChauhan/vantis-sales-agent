"""T13e: the fixes from Adithya's live test of 562f27e (14 Oct), from his examples plus variants. The understand step's
output is stubbed (the AI understands); these test what the code enforces."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import welcome
from app.agent.fallback import closing_question, fallback_reply
from app.agent.graph import tool_outcome_reply
from app.agent.leads import (
    INFO_ONLY_LINE, MAX_SKIPPED_ASKS, PROBLEM_UNCLEAR_LINE, SAVED_EXPLORING_LINE, LeadOps, status_block,
)
from app.agent.packages import add_offerings_line, add_size_line
from app.agent.questions import ask_rules, fix_region_reply, offer_kind, offers_made
from app.leads.reason import problem_with
from app.rag.answer import add_missing_tag, plain_kb_words, remove_kb_rules, strip_unsourced_citations
from app.rag.retriever import RetrievedChunk
from app.services.service_areas import ServiceAreas
from app.rag.retriever import SearchResult
from tests.agent_fakes import CATALOG_SUMMARY, FakeSearch, Scripted, answer, make_graph, run, understood

# A service-area table with KB Section 10's shape and its five regions (test data, like tests/leads_data.py AREAS)
REGION_AREAS = ServiceAreas.from_rows([
    {"Region": "Uttar Pradesh", "Reference Locations": "Lucknow, Kanpur, Noida"},
    {"Region": "Delhi NCR", "Reference Locations": "Delhi, Gurugram, Noida"},
    {"Region": "Punjab", "Reference Locations": "Ludhiana, Amritsar"},
    {"Region": "Haryana", "Reference Locations": "Panipat, Karnal"},
    {"Region": "Rajasthan", "Reference Locations": "Jaipur, Kota"},
])
NO_DB = LeadOps(engine=None, areas_fn=lambda: REGION_AREAS)
AREAS_CHUNK = RetrievedChunk("s10-table-summary", 10, "Locations", 3, "table_summary",
                             "All rows (Region; Reference Locations):\n- Delhi NCR; Delhi, Gurugram, Noida")


# ---------- 1. Regions are not cities ----------

REGION_MESSAGES = [  # (message, the name as the understand step writes it, the KB region), English and Hinglish
    ("I'm in Delhi NCR. Do you cover my area?", "Delhi NCR", "Delhi NCR"),
    ("Delhi NCR mein service hai kya?", "Delhi NCR", "Delhi NCR"),
    ("Do you install in NCR?", "NCR", "Delhi NCR"),
    ("NCR me lagate ho?", "NCR", "Delhi NCR"),
    ("I live in Uttar Pradesh", "Uttar Pradesh", "Uttar Pradesh"),
    ("main Uttar Pradesh se hu", "Uttar Pradesh", "Uttar Pradesh"),
    ("Do you work in UP?", "UP", "Uttar Pradesh"),
    ("UP mein solar lagwate ho kya?", "UP", "Uttar Pradesh"),
    ("Do you cover Punjab?", "Punjab", "Punjab"),
    ("Punjab mein installation karte ho?", "Punjab", "Punjab"),
    ("I'm from Haryana, can I get solar?", "Haryana", "Haryana"),
    ("Haryana me solar lagwana hai", "Haryana", "Haryana"),
    ("Is Rajasthan covered?", "Rajasthan", "Rajasthan"),
    ("Rajasthan se hu, aap aate ho?", "Rajasthan", "Rajasthan"),
]


@pytest.mark.parametrize("message, name, region", REGION_MESSAGES)
def test_a_region_is_checked_as_a_region_and_never_stored(message, name, region):
    """Adithya: "I'm in Delhi NCR. Do you cover my area?" got "Vantis currently covers Delhi NCR" (area_check:
    listed)."""
    agent = Scripted(answer(f"Yes, Vantis covers {name} [KB Section 10: Locations]."))
    graph, _ = make_graph(Scripted(understood(["question", "buying_interest"], "Reference Service Locations",
                                              cities=[name], place_relation="region")), agent, ops=NO_DB,
                          areas=REGION_AREAS)
    state = run(graph, message)
    assert [(c["city"], c["status"]) for c in state["area_checks"]] == [(name, "region")]
    assert "city" not in state["lead"] and state["asked_cities"] == []
    cities = REGION_AREAS.regions[region]
    assert f"Vantis covers {name}" not in state["reply"]
    line = welcome.AREA_REGION_TEXT.format(region=region, cities=_join(cities))
    assert f"{line[:-1]} [KB Section 10: Locations]." in state["reply"]
    assert state["reply"].endswith(welcome.REGION_CITY_ASK.format(region=region))
    assert "region_cities" in str(agent.calls[0][-1].content)  # the model reads the region's cities


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def test_a_region_in_the_city_field_is_never_stored():
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, city="Punjab", place_relation="region")),
                          Scripted(answer("Thanks.")), ops=NO_DB, areas=REGION_AREAS)
    state = run(graph, "main Punjab se hu")
    assert "city" not in state["lead"] and state["area_checks"][0]["status"] == "region"


def test_a_place_the_model_calls_a_region_that_is_a_city_is_never_confirmed():
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, city="Delhi", place_relation="region")),
                          Scripted(answer("Thanks.")), ops=NO_DB, areas=REGION_AREAS)
    state = run(graph, "Delhi side")
    assert "city" not in state["lead"] and state["area_checks"][0]["status"] == "unverified"


def test_a_region_found_in_the_words_when_understand_fails_is_never_stored():
    graph, _ = make_graph(Scripted(RuntimeError("down")), Scripted(answer("Thanks.")), ops=NO_DB, areas=REGION_AREAS)
    state = run(graph, "main Punjab se hu, solar lagwana hai")
    assert "city" not in state["lead"] and [(c["city"], c["status"]) for c in state["area_checks"]] == [
        ("Punjab", "region")]


def test_naming_a_region_when_asked_for_the_city_answers_it():
    graph, _ = make_graph(Scripted(understood(["buying_interest"], None, customer_type="residential"),
                                   understood(["gives_details"], None, cities=["NCR"], place_relation="region")),
                          Scripted(answer("Great! Which city is the property in?"),
                                   answer("Vantis's service areas there are Delhi, Gurugram and Noida.")),
                          ops=NO_DB, areas=REGION_AREAS)
    run(graph, "I want solar for my home", thread="r")
    assert run(graph, "NCR", thread="r")["skipped_asks"] == 0


def region_state(**extra) -> dict:
    check = {"city": "Delhi NCR", "status": "region", "regions": ["Delhi NCR"],
             "region_cities": ["Delhi", "Gurugram", "Noida"]}
    return {"lead": {}, "area_checks": [check], "messages": [HumanMessage("I'm in Delhi NCR")], **extra}


def test_the_region_guard_keeps_a_reply_that_names_the_cities_and_asks():
    reply = ("Vantis's service areas in Delhi NCR are Delhi, Gurugram and Noida [KB Section 10: Locations]. Which city "
             "is your property in?")
    assert fix_region_reply(reply, region_state(), [AREAS_CHUNK])[0] == reply  # its statement is the fixed line


LINE = "Vantis's service areas in Delhi NCR are Delhi, Gurugram and Noida [KB Section 10: Locations]."
ASK = welcome.REGION_CITY_ASK.format(region="Delhi NCR")


@pytest.mark.parametrize("reply, expected", [
    # the two live misses (F2 on the main model, F1 on the fast one): a coverage claim that also lists the cities
    ("Vantis covers the Delhi NCR area, which includes the service areas of Noida, Delhi and Gurugram "
     "[KB Section 10: Locations]. Which city is your property in?", f"{LINE} Which city is your property in?"),
    ("Yes, we serve the Delhi NCR area, which includes Delhi, Gurugram and Noida [KB Section 10: Locations].\n\n"
     "Which city is your property located in?", f"{LINE}\n\nWhich city is your property located in?"),
    # Adithya's other phrasings: every statement of ours naming the region or its short form is replaced
    ("Delhi NCR is within our service area.", f"{LINE} {ASK}"),
    ("We cover the NCR.", f"{LINE} {ASK}"),
    # an opening "Yes" is dropped, whatever follows it
    ("Yes! Happy to help.", f"{LINE}\n\nHappy to help. {ASK}"),
    ("Yes, Delhi is within our current service areas [KB Section 10: Locations].", f"{LINE} {ASK}"),
    # a question naming the region stays (the one-question guard runs next)
    ("Which part of Delhi NCR is your home in?", f"{LINE}\n\nWhich part of Delhi NCR is your home in? {ASK}"),
])
def test_every_statement_naming_the_region_becomes_the_fixed_line(reply, expected):
    assert fix_region_reply(reply, region_state(), [AREAS_CHUNK])[0] == expected


def test_a_lowercase_up_in_our_reply_is_never_uttar_pradesh():
    check = {"city": "UP", "status": "region", "regions": ["Uttar Pradesh"], "region_cities": ["Lucknow", "Kanpur"]}
    state = {"lead": {}, "area_checks": [check], "messages": [HumanMessage("UP mein solar lagwate ho kya?")]}
    text, _ = fix_region_reply("We can set up a site visit once we know your city.", state, [])
    assert "We can set up a site visit once we know your city." in text


def test_the_region_guard_replaces_a_claim_about_your_area():
    text, changed = fix_region_reply("Yes, we serve your area! Would you like a callback?", region_state(),
                                     [AREAS_CHUNK])
    assert text.startswith("Vantis's service areas in Delhi NCR are Delhi, Gurugram and Noida "
                           "[KB Section 10: Locations].")
    assert "we serve your area" not in text and changed[-1] == "asked which city"


def test_the_region_guard_asks_nothing_while_asks_are_parked():
    text, _ = fix_region_reply("Delhi NCR is a big region.", region_state(skipped_asks=MAX_SKIPPED_ASKS), [])
    assert "?" not in text and "Delhi, Gurugram and Noida" in text


def test_the_safety_net_names_a_regions_cities_and_asks_which_city():
    state = {**region_state(), "search_query": None, "new_details": []}
    reply, _, _ = fallback_reply(state, [AREAS_CHUNK.__dict__], None)
    assert "Vantis's service areas in Delhi NCR are Delhi, Gurugram and Noida [KB Section 10: Locations]." in reply
    assert reply.endswith(welcome.REGION_CITY_ASK.format(region="Delhi NCR"))


# ---------- 2. The same question four times ----------

LEADING = ["5kW ka total kharcha 2 lakh tak aa jayega na?", "subsidy 30% milti hai na?",
           "ek mahine me installation pakka ho jayega?", "bill 70% kam ho jayega na?"]


def test_leading_questions_count_as_skips_and_the_city_is_never_asked_four_times():
    """Adithya's live test: "Which city is your property located in?" closed four replies in a row; numbers inside the
    customer's questions reset the count. They are asks_new now, whatever the extractor read from them."""
    ask = "Which city is your property located in?"
    agent = Scripted(answer(f"Great choice! {ask}"), *[answer(f"That depends on the site. {ask}") for _ in LEADING])
    understandings = [understood(["buying_interest"], None, customer_type="residential"),
                      understood(["question"], "Pricing", reply_type="asks_new", system_size_kw="5 kW",
                                 budget_inr="2 lakh"),
                      understood(["question"], "Subsidies", reply_type="asks_new"),
                      understood(["question"], "Installation timing", reply_type="asks_new", timeline="ek mahine",
                                 timeline_bucket="lt_1m"),
                      understood(["question"], "Savings", reply_type="asks_new")]
    graph, _ = make_graph(Scripted(*understandings), agent, ops=NO_DB)
    run(graph, "mujhe ghar pe solar lagwana hai", thread="lead")
    replies = [run(graph, message, thread="lead") for message in LEADING]
    # a removed repeat leaves the reply with nothing asked, so the next message skips nothing; then asks are parked
    assert [r["skipped_asks"] for r in replies] == [1, 1, 2, 2]
    assert [ask in r["reply"] for r in replies] == [False, True, False, False]  # never twice in a row


@pytest.mark.parametrize("previous, question, removed", [
    ("Which city is your property located in?", "Which city is your property located in?", True),
    ("Which city is your property located in?", "Which city is the property located in?", True),  # 91
    ("Could you share your name and phone number?", "Could you share your phone number?", False),  # 88: a follow-up
    ("Could you share your name?", "Could you share your phone number?", False),
])
def test_a_question_that_repeats_the_last_one_is_removed(previous, question, removed):
    state = {"lead": {"customer_type": "residential"}, "intents": ["buying_interest"],
             "messages": [AIMessage(f"Thanks. {previous}"), HumanMessage("hmm")]}
    text, changed = ask_rules(f"Sure. {question}", state)
    assert (question not in text) is removed and bool(changed) is removed


def test_a_validation_error_may_ask_again():
    state = {"lead": {}, "intents": ["gives_details"], "validation_errors": {"phone": "looks too short"},
             "messages": [AIMessage("Could you share your phone number?"), HumanMessage("12345")]}
    assert ask_rules("That looks short. Could you share your phone number?", state)[1] == []


# ---------- 3. The same offer every turn ----------

OFFER = "Would you like to schedule a site assessment to discuss your options?"


def test_the_same_offer_appears_once_per_chat_even_reworded():
    history = [HumanMessage("how long?"), AIMessage(f"Timing varies. {OFFER}"), HumanMessage("warranty?")]
    state = {"lead": {}, "intents": ["question"], "messages": history + [HumanMessage("subsidy?")]}
    assert offers_made(history) == ["site assessment"]
    text, changed = ask_rules(f"Warranty depends on the equipment. {OFFER}", state)
    assert text == "Warranty depends on the equipment." and changed  # word for word: a repeat of the last question
    text, changed = ask_rules("Warranty depends on the equipment. Shall we arrange a site assessment for your home?",
                              state)
    assert text == "Warranty depends on the equipment."
    assert changed[0]["why"] == "the site assessment was offered before"
    text, _ = ask_rules("Warranty depends on the equipment. Would you like the team to send a quote?", state)
    assert text.endswith("send a quote?")  # a different kind of offer is new


def test_offer_kinds():
    assert offer_kind(OFFER) == "site assessment"
    assert offer_kind("Would you like a callback from the team?") == "callback"
    assert offer_kind("Shall I check whether Vantis covers your city?") == "city check"
    assert offer_kind("Would you like to share your monthly bill?") is None  # asks for a detail
    assert offer_kind("Which city is the property in?") is None


def test_the_status_block_lists_the_offers_made():
    assert "Offers already made in this chat: site assessment. Do not make them again." in status_block(
        {"lead": {}}, ["site assessment"])


# ---------- 4. Not pushy (KB Section 17) ----------

EXPLORER = {"lead": {"name": "Vikram", "phone": "+919876500004", "city": "Kanpur", "intent_level": "exploring",
                     "customer_type": "residential"}, "lead_id": 4, "lead_created_now": True,
            "messages": [HumanMessage("just researching")]}


def test_an_explorer_is_never_asked_for_a_callback_time():
    text, changed = ask_rules("Your details are saved. What time would suit you for a callback?", EXPLORER)
    assert text == f"Your details are saved.\n\n{welcome.EXPLORER_READY_TEXT}"
    assert changed[0]["why"] == "an explorer is never asked about a callback"
    assert SAVED_EXPLORING_LINE in status_block(EXPLORER)
    scratch = [AIMessage("")]
    assert tool_outcome_reply(scratch, [{"kind": "lead", "created": True}], exploring=True) == (
        welcome.SAVED_EXPLORING_TEXT)


def test_a_ready_customer_still_gets_the_callback_ask():
    ready = {**EXPLORER, "lead": {**EXPLORER["lead"], "intent_level": "ready"}}
    assert ask_rules("Your details are saved. What time would suit you for a callback?", ready)[1] == []
    assert tool_outcome_reply([], [{"kind": "lead", "created": True}]) == welcome.SAVED_FALLBACK_TEXT


INFO_ONLY = {"lead": {"city": "Delhi", "intent_level": "interested"}, "city_source": "asked", "intents": ["question"],
             "messages": [HumanMessage("Do you provide installation in Delhi?")]}


def test_an_info_only_question_about_a_city_asks_for_no_contact_details():
    """Adithya: after "Do you provide installation in Delhi?" the bot asked for name and phone."""
    assert INFO_ONLY_LINE in status_block(INFO_ONLY)
    text, changed = ask_rules("Vantis currently covers Delhi. Could you share your name and phone number?", INFO_ONLY)
    assert text == "Vantis currently covers Delhi." and "info-only" in changed[0]["why"]


def test_one_detail_per_question_but_name_and_phone_may_pair():
    state = {"lead": {"customer_type": "residential"}, "intents": ["buying_interest"],
             "messages": [HumanMessage("I want solar for my home")]}
    text, changed = ask_rules("Great! Could you share your city, name and phone number?", state)
    assert text == f"Great! {welcome.NEXT_QUESTION['city']}" and changed[0]["why"].startswith("more than one")
    contact = {"lead": {}, "intents": ["contact_request"], "contact_request": True,
               "messages": [HumanMessage("Can someone call me?")]}
    assert ask_rules("Sure! Could you share your name and phone number?", contact)[1] == []


# ---------- 5. No KB tags on "I don't know" sentences ----------

S1 = RetrievedChunk("s01", 1, "Company Overview", 2, "prose",
                    "Vantis provides end-to-end solar solutions including rooftop installation and system sizing.")
S11 = RetrievedChunk("s11-panels", 11, "Warranty", 4, "table_row", "Component / Area: Solar Panels\nReference Policy: "
                     "Warranty is governed by the applicable manufacturer and project documentation.")


@pytest.mark.parametrize("reply, chunks, expected", [
    ("I don't have information on an exact installation timeline [KB Section 1: Company Overview].", [S1],
     "I don't have information on an exact installation timeline."),
    ("I can't confirm any cashback offers [KB Section 11: Warranty].", [S11], "I can't confirm any cashback offers."),
    ("I don't have information on panel cleaning. [KB Section 11: Warranty]", [S11],
     "I don't have information on panel cleaning."),
    ("Panel warranty is governed by the manufacturer documentation, but I can't confirm the exact duration "
     "[KB Section 11: Warranty].", [S11], None),  # it also states the section's fact: the tag stays
])
def test_an_i_dont_know_sentence_keeps_no_unrelated_tag(reply, chunks, expected):
    assert strip_unsourced_citations(reply, chunks)[0] == (expected or reply)


def test_a_missing_tag_is_never_added_to_an_i_dont_know_sentence():
    reply = "I don't have information on rooftop installation sizing for solar systems."
    assert add_missing_tag(reply, [S1]) == (reply, None)


# ---------- 6. Wording ----------

def test_never_the_knowledge_base_to_the_customer():
    assert plain_kb_words("Pune isn't listed in the knowledge base yet [KB Section 10: Locations].")[0] == (
        "Pune isn't listed in our current service list yet [KB Section 10: Locations].")
    assert plain_kb_words("The knowledge base doesn't mention a cleaning schedule.")[0] == (
        "The information I have doesn't mention a cleaning schedule.")
    assert plain_kb_words("Warranty depends on the equipment [KB Section 11: Warranty].")[1] == 0


@pytest.mark.parametrize("reply, expected", [
    ("Panel warranty is governed by the manufacturer and project documentation, and no universal duration should be "
     "assumed [KB Section 11: Warranty].",
     "Panel warranty is governed by the manufacturer and project documentation [KB Section 11: Warranty]."),
    ("Do not assume 5 kW suits every home, but HomePro 5 is the 5 kW reference package.",
     "HomePro 5 is the 5 kW reference package."),
    ("No universal duration should be assumed. Warranty depends on the equipment.",
     "Warranty depends on the equipment."),
    ("You should check eligibility with the team.", "You should check eligibility with the team."),
])
def test_a_kb_rule_clause_is_cut_and_the_fact_kept(reply, expected):
    assert remove_kb_rules(reply)[0] == expected


def test_the_size_line_for_a_stated_size():
    """Adithya: "I'm interested in installing a 5kW solar system for my house" named HomePro 5 without saying the site
    assessment confirms the final size (KB Section 9)."""
    agent = Scripted(answer("Plan B is our 5 kW reference package for homes [KB Section 8: Catalog]. Would you like to "
                            "know more?"))
    graph, _ = make_graph(Scripted(understood(["buying_interest", "gives_details"], None, system_size_kw="5 kW",
                                              customer_type="residential")), agent)
    state = run(graph, "I'm interested in installing a 5kW solar system for my house")
    assert state["chunks"][0] == CATALOG_SUMMARY
    assert f"{welcome.SIZE_ASSESSMENT_TEXT[:-1]} [KB Section 8: Catalog]." in state["reply"]
    assert add_size_line("A site visit decides the size; Plan B is 5 kW.", {"home": ["Plan A", "Plan B"]},
                         welcome.SIZE_ASSESSMENT_TEXT)[1] is False


# ---------- 7. The lead reason never overclaims the service area ----------

@pytest.mark.parametrize("part", [
    "WARM: a ready buyer, but the location is outside our service area",
    "COLD: the city is not in our service area",
    "WARM: interested, though we don't serve Pune",
    "WARM: interested, but Vantis doesn't cover that city",
    "WARM: the city is out of our service area",
    "WARM: the location is not served",
])
def test_a_reason_that_overclaims_the_service_area_gets_the_template(part):
    status = part.split(":")[0]
    assert problem_with(part.split(": ", 1)[1], status) == "overclaims the service area"


def test_the_listed_service_areas_wording_still_passes():
    assert problem_with("ready to buy, city not in the listed service areas", "WARM") is None


# ---------- the safety net never repeats the last question ----------

def test_the_safety_nets_question_never_repeats_the_last_one():
    state = {"lead": {}, "intents": ["question"], "messages": [AIMessage(f"Timing varies. {welcome.ANYTHING_ELSE_ASK}"),
                                                               HumanMessage("ok and warranty?")],
             "skipped_asks": 1}
    assert closing_question(state) == (None, {})
    state = {**state, "skipped_asks": 0}
    assert closing_question(state) == (welcome.FALLBACK_OFFER, {"fallback_offered": True})


# ---------- round 2: Adithya's three live failures (the AI's flags, checks on our own reply, KB names only) ----------

@pytest.mark.parametrize("problem_now, support", [("what_if", False), ("unsure", False), ("no", False), ("yes", True)])
def test_only_a_problem_now_opens_the_support_flow(problem_now, support):
    """R2: "agar inverter kharab ho jaye to kya karna padega" was labelled a support issue. The understand step's
    problem_now decides: only "yes" starts the support flow; an unclear one gets the general answer and one question."""
    agent = Scripted(answer("Vantis can assist with maintenance and support requests."))
    graph, _ = make_graph(Scripted(understood(["question", "existing_customer_issue"], "Reference Support Scenarios",
                                              issue="inverter fault", problem_now=problem_now)), agent, ops=NO_DB)
    state = run(graph, "a message about the inverter")
    assert state["support_case"] is support and ("existing_customer_issue" in state["intents"]) is support
    assert (PROBLEM_UNCLEAR_LINE in str(agent.calls[0][-1].content)) is (problem_now == "unsure")
    assert state["steps"][0]["data"]["problem_now"] == problem_now


def test_yes_without_an_issue_is_unclear():
    """F11 in the T13e rerun: "I have a question about my inverter" came back as a problem now. "yes" needs the problem
    itself (the AI's issue field, from this message or an earlier one); without it the turn is treated as unsure."""
    agent = Scripted(answer("Vantis can assist with maintenance and support requests."))
    graph, _ = make_graph(Scripted(understood(["question", "existing_customer_issue"], "Reference Support Scenarios",
                                              problem_now="yes")), agent, ops=NO_DB)
    state = run(graph, "I have a question about my inverter")
    assert state["support_case"] is False and state["steps"][0]["data"]["problem_now"] == "unsure"
    assert PROBLEM_UNCLEAR_LINE in str(agent.calls[0][-1].content)


def test_yes_with_the_issue_from_an_earlier_message_stays_yes():
    graph, _ = make_graph(Scripted(understood(["existing_customer_issue"], None, issue="inverter shows a red light"),
                                   understood(["existing_customer_issue"], None, problem_now="yes")),
                          Scripted(answer("Sorry to hear that.")), ops=NO_DB)
    run(graph, "my inverter shows a red light", thread="s")
    assert run(graph, "it's still not working", thread="s")["steps"][0]["data"]["problem_now"] == "yes"


def test_a_problem_now_opens_the_support_flow_even_without_the_intent():
    graph, _ = make_graph(Scripted(understood(["question"], "inverter", problem_now="yes", issue="no power")),
                          Scripted(answer("Sorry to hear that.")), ops=NO_DB)
    assert run(graph, "a message about the inverter")["support_case"] is True


def test_the_safety_net_asks_whether_something_is_wrong_now():
    state = {"lead": {}, "problem_unclear": True, "messages": [HumanMessage("I have a question about my inverter")]}
    assert closing_question(state) == (welcome.PROBLEM_NOW_ASK, {})


S1_OVERVIEW = RetrievedChunk("s01", 1, "Company Overview", 2, "prose",
                             "Vantis provides end-to-end solar solutions including rooftop installation.")


def test_what_do_you_offer_gets_the_catalog_first_and_always_names_the_offerings():
    """R1, the assignment's first example question: the reply came from KB Section 1's services and named no package.
    The understand step's asks_offerings puts the KB Section 8 list first, and a reply naming none of its offerings gets
    the list from the KB with its tag (a check on our own reply)."""
    agent = Scripted(answer("Vantis provides end-to-end solar solutions [KB Section 1: Company Overview]. Would you "
                            "like to check if we cover your city?"))
    graph, _ = make_graph(Scripted(understood(["question"], "Company Overview solar solutions", asks_offerings=True)),
                          agent, search=FakeSearch(SearchResult(chunks=[S1_OVERVIEW], mode="hybrid")))
    state = run(graph, "What solar systems do you provide?")
    assert state["chunks"][0] == CATALOG_SUMMARY
    assert ("Vantis's offerings are:\n1. Plan A (Residential)\n2. Plan B (Residential) [KB Section 8: Catalog]"
            in state["reply"])
    assert state["reply"].endswith("Would you like to check if we cover your city?")
    assert {s["section"] for s in state["sources"]} == {"1", "8"}


def test_a_reply_that_names_the_offerings_is_left_alone_and_other_questions_get_no_list():
    catalog = ("All rows (Offering; Primary Customer; Reference Profile):\n- Vantis HomeStart 3; Residential; 3 kW\n"
               "- Vantis Care; Existing Customers; support")
    for reply in ("We offer HomeStart 3 and Vantis Care [KB Section 8: Catalog].",  # with or without the brand
                  "Vantis HomeStart 3 suits moderate household use."):
        assert add_offerings_line(reply, catalog, welcome.OFFERINGS_INTRO) == (reply, False)
    added, listed = add_offerings_line("We do rooftop solar.", catalog, welcome.OFFERINGS_INTRO, "[KB Section 8: X]")
    assert listed and added.endswith("1. Vantis HomeStart 3 (Residential)\n2. Vantis Care (Existing Customers) "
                                     "[KB Section 8: X]")
    agent = Scripted(answer("Pricing depends on the project."))
    graph, _ = make_graph(Scripted(understood(["question"], "Pricing")), agent)
    assert "offerings are" not in run(graph, "How much does solar cost?")["reply"]


def test_a_city_used_only_inside_a_region_name_is_the_region():
    """E1: the fallback model wrote "Delhi" for "I'm in Delhi NCR". A listed name the message uses only inside a KB
    region's name is that region; it matches KB names only, so it can stop a confirmation, never make one."""
    agent = Scripted(answer("Yes, Delhi is within our current service areas [KB Section 10: Locations]."))
    graph, _ = make_graph(Scripted(understood(["question"], "Reference Service Locations", city="Delhi")), agent,
                          ops=NO_DB, areas=REGION_AREAS)
    state = run(graph, "I'm in Delhi NCR. Do you cover my area?")
    assert "city" not in state["lead"]
    assert [(c["city"], c["status"]) for c in state["area_checks"]] == [("Delhi NCR", "region")]
    assert "Vantis's service areas in Delhi NCR are Noida, Delhi and Gurugram" in state["reply"]  # table order
    assert state["reply"].endswith(welcome.REGION_CITY_ASK.format(region="Delhi NCR"))
    graph, _ = make_graph(Scripted(understood(["question"], "Reference Service Locations", city="Delhi")),
                          Scripted(answer("Vantis currently covers Delhi.")), ops=NO_DB, areas=REGION_AREAS)
    state = run(graph, "I live in Delhi itself, not elsewhere in NCR")
    assert state["lead"]["city"] == "Delhi" and state["area_checks"][0]["status"] == "listed"


@pytest.mark.parametrize("text, name, region", [
    ("I'm in Delhi NCR. Do you cover my area?", "Delhi", "Delhi NCR"),
    ("Mera office Delhi NCR mein hai", "Delhi", "Delhi NCR"),
    ("NCR Delhi side", "Delhi", "Delhi NCR"),
    ("I live in Delhi", "Delhi", None),
    ("Delhi NCR side, but the house is in Delhi city", "Delhi", None),  # Delhi on its own too: the AI's city stays
    ("I'm in Lucknow, UP", "Lucknow", None),
])
def test_region_around_reads_only_kb_names(text, name, region):
    assert REGION_AREAS.region_around(text, name) == region

"""T13e: the understand step says what a reply does with our last message (reply_type); the code routes, counts skipped
questions and parks asks and offers from it, never from the customer's words. The word lists run only when the
understand step failed (the backup), so they are tested with the understand step raising."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import welcome
from app.agent.leads import MAX_SKIPPED_ASKS, LeadOps, ask_text, _still_needed_ask
from app.agent.questions import last_question_kind
from app.agent.text_details import reply_type_from_words, short_reply
from app.agent.understand import HISTORY_CHARS, KbHints, build_prompt
from app.core.llm import LLMError
from tests.agent_fakes import Scripted, answer, make_graph, run, understood
from tests.leads_data import AREAS

NO_DB = LeadOps(engine=None, areas_fn=lambda: AREAS)  # lead tools on; nothing gets ready to save in these tests
OFFER = "Timing depends on the site [KB Section 15: FAQ]. Would you like to schedule a site assessment?"
CITY_ASK = "Great choice! Which city is the property in?"
ROOF_ASK = "Thanks! Does the property have usable roof space for the panels?"
CONTACT_ASK = "Sure. Could you share your name and phone number?"
NOTHING = "Timing depends on the site [KB Section 15: FAQ]."
OPEN_QUESTIONS = ["Anything else I can help with?", "Is there anything else I can help you with?",
                  "Can I help with anything else?", "Any other questions?"]
FIRST = understood(["buying_interest"], None, customer_type="residential")  # the customer shows interest


def chat(first_reply: str, second, second_reply: str = "Okay, noted your question.", first=FIRST,
         message: str = "next message"):
    """(the state after turn 2, the scripted agent): turn 1's reply is `first_reply`; turn 2 sends `message`, and its
    understand output is `second` (a stub, or an exception for a failed understand step)."""
    agent = Scripted(answer(first_reply), answer(second_reply))
    graph, _ = make_graph(Scripted(first, second), agent, ops=NO_DB)
    run(graph, "I want rooftop solar for my home", thread="rt")
    return run(graph, message, thread="rt"), agent


# ---------- the routing, one value at a time (the understand output stubbed) ----------

def test_answers_resets_the_count():
    state, _ = chat(CITY_ASK, understood(["gives_details"], None, city="Kanpur"))
    assert state["skipped_asks"] == 0


def test_skips_counts_after_a_detail_question_but_not_after_a_reply_that_asked_nothing():
    state, agent = chat(CITY_ASK, understood(["reply_to_assistant"], None, reply_type="skips"))
    assert state["skipped_asks"] == 1 and len(agent.calls) == 2  # the model still answers
    state, _ = chat(NOTHING, understood(["reply_to_assistant"], None, reply_type="skips"))
    assert state["skipped_asks"] == 0


def test_accepts_goes_to_the_model_and_resets_the_count():
    state, agent = chat(OFFER, understood(["reply_to_assistant"], None, reply_type="accepts"))
    assert len(agent.calls) == 2 and state["reply"] != welcome.CLOSING_TEXT and state["skipped_asks"] == 0


def test_declines_an_offer_parks_asks_and_offers_at_once():
    agent = Scripted(answer(OFFER), answer("Okay."), answer("Warranty depends on the equipment. Would you like a "
                                                            "callback from the team?"))
    graph, _ = make_graph(Scripted(FIRST, understood(["reply_to_assistant"], None, reply_type="declines"),
                                   understood(["question"], "warranty")), agent, ops=NO_DB)
    run(graph, "I want rooftop solar for my home", thread="no")
    assert run(graph, "no thanks", thread="no")["skipped_asks"] == MAX_SKIPPED_ASKS
    third = run(graph, "what warranty do you give?", thread="no")
    assert third["reply"] == "Warranty depends on the equipment."  # no offer while parked
    assert any(c["why"] == "no offers after skipped or declined ones"
               for s in third["steps"] if s["node"] == "guard" for c in s["data"].get("changed", []))


def test_goes_ahead_brings_the_asks_back():
    agent = Scripted(answer(OFFER), answer("Okay."), answer("Great! Which city is the property in?"))
    graph, _ = make_graph(Scripted(FIRST, understood(["reply_to_assistant"], None, reply_type="declines"),
                                   understood(["buying_interest"], None, reply_type="goes_ahead")), agent, ops=NO_DB)
    run(graph, "I want rooftop solar for my home", thread="go")
    run(graph, "no thanks", thread="go")
    third = run(graph, "actually let's book it", thread="go")
    assert third["skipped_asks"] == 0 and third["reply"].endswith("Which city is the property in?")


def test_closes_gives_the_closing_text_without_a_model_call():
    state, agent = chat(NOTHING, understood(["thanks_or_bye"], None, reply_type="closes"))
    assert state["reply"] == welcome.CLOSING_TEXT and len(agent.calls) == 1
    assert state["steps"][0]["data"]["closes"] is True


def test_closes_never_drops_a_question_or_a_detail():
    """Adithya: "closes" only when the message does nothing else; the code checks the other fields agree."""
    state, agent = chat(NOTHING, understood(["reply_to_assistant", "question"], "warranty", reply_type="closes"))
    assert len(agent.calls) == 2 and state["reply"] != welcome.CLOSING_TEXT
    state, agent = chat(NOTHING, understood(["thanks_or_bye"], None, reply_type="closes", monthly_bill_inr="4000"))
    assert len(agent.calls) == 2 and state["lead"]["monthly_bill_inr"] == 4000


def test_asks_new_counts_a_skip_and_the_model_answers():
    """Adithya's live chat: leading questions with numbers ("5kW ka total kharcha 2 lakh tak aa jayega na?") reset the
    count; asks_new counts whatever details the extractor read from them."""
    state, agent = chat(CITY_ASK, understood(["question"], "Pricing", reply_type="asks_new", system_size_kw="5 kW"))
    assert state["skipped_asks"] == 1 and len(agent.calls) == 2


def test_the_understand_step_runs_on_a_bare_ok_and_sees_the_last_reply_in_full():
    understand_llm = Scripted(FIRST, understood(["reply_to_assistant"], None, reply_type="accepts"))
    graph, _ = make_graph(understand_llm, Scripted(answer(OFFER), answer("Great.")), ops=NO_DB)
    run(graph, "I want rooftop solar for my home", thread="seen")
    run(graph, "ok", thread="seen")
    assert len(understand_llm.calls) == 2
    long_reply = "Here is the whole list. " * 40 + "Which city is the property in?"
    prompt = build_prompt(KbHints(), "ok", [HumanMessage("hi"), AIMessage(long_reply)])
    assert len(long_reply) > HISTORY_CHARS and prompt.count("Which city is the property in?") == 1  # only its block


# ---------- declines stays narrow (Adithya's two chats) ----------

def test_a_no_that_answers_a_question_never_parks():
    """"Do you own the house?" -> "No, it's rented." is an answer; even a model that labels it declines parks
    nothing, since the question was neither an offer nor a request for contact details."""
    for label in ("answers", "declines"):
        state, _ = chat(ROOF_ASK, understood(["gives_details"], None, rooftop_available=False, reply_type=label))
        assert state["skipped_asks"] == 0


def test_refusing_one_detail_rules_it_out_and_asks_the_other():
    state, _ = chat(CONTACT_ASK, understood(["gives_details"], None, reply_type="answers",
                                            ruled_out=["phone number"]),
                    first=understood(["contact_request"], None))
    assert state["skipped_asks"] == 0 and state["ruled_out"] == ["phone number"]
    assert ask_text("phone or email", state) == welcome.NEXT_QUESTION["email"]
    assert "email address" in _still_needed_ask({"lead": {"name": "Asha"}, "contact_request": True,
                                                 "ruled_out": ["phone number"]})
    # a decline label with a ruled-out detail still doesn't park: the customer offered another way
    state, _ = chat(CONTACT_ASK, understood(["reply_to_assistant"], None, reply_type="declines",
                                            ruled_out=["phone number"]),
                    first=understood(["contact_request"], None))
    assert state["skipped_asks"] == 0


def test_a_clear_no_to_an_offer_parks():
    state, _ = chat("Would you like a callback from the team?", understood(["reply_to_assistant"], None,
                                                                          reply_type="declines"))
    assert state["skipped_asks"] == MAX_SKIPPED_ASKS


# ---------- the backup: the understand step failed, the words decide ----------

DOWN = LLMError("main and fast down")


@pytest.mark.parametrize("message", ["ok", "okay", "fine", "theek hai", "yes thanks", "sure, thank you"])
def test_backup_ok_or_yes_after_an_offer_is_a_yes(message):
    state, agent = chat(OFFER, DOWN, message=message)
    assert len(agent.calls) == 2 and state["skipped_asks"] == 0 and state["reply"] != welcome.CLOSING_TEXT
    assert state["steps"][0]["detail"] == "failed, raw message used"


@pytest.mark.parametrize("message", ["ok", "ok thanks", "okay, thank you", "fine thanks", "thanks 🙏", "thx"])
def test_backup_ok_or_thanks_after_a_reply_that_asked_nothing_closes(message):
    agent = Scripted(answer(NOTHING), answer("unused"))
    graph, _ = make_graph(Scripted(FIRST, DOWN), agent, ops=NO_DB)
    run(graph, "I want rooftop solar for my home", thread="b1")
    assert run(graph, message, thread="b1")["reply"] == welcome.CLOSING_TEXT and len(agent.calls) == 1


@pytest.mark.parametrize("message", ["ok thanks", "okay, thank you", "fine thanks"])
def test_backup_ok_with_thanks_closes_even_after_an_offer(message):
    agent = Scripted(answer(OFFER), answer("unused"))
    graph, _ = make_graph(Scripted(FIRST, DOWN), agent, ops=NO_DB)
    run(graph, "I want rooftop solar for my home", thread="b2")
    assert run(graph, message, thread="b2")["reply"] == welcome.CLOSING_TEXT


def test_backup_bare_ok_after_a_detail_question_is_a_skip_not_a_goodbye():
    state, agent = chat(CITY_ASK, DOWN, message="ok")
    assert state["skipped_asks"] == 1 and len(agent.calls) == 2 and state["reply"] != welcome.CLOSING_TEXT


@pytest.mark.parametrize("question", OPEN_QUESTIONS)
def test_backup_bare_ok_after_an_open_question_closes(question):
    agent = Scripted(answer(f"Warranty depends on the equipment. {question}"), answer("unused"))
    graph, _ = make_graph(Scripted(FIRST, DOWN), agent, ops=NO_DB)
    run(graph, "I want rooftop solar for my home", thread="b3")
    assert run(graph, "ok", thread="b3")["reply"] == welcome.CLOSING_TEXT


@pytest.mark.parametrize("message", ["no thanks", "not interested", "nahi chahiye"])
def test_backup_a_clear_no_after_an_offer_parks(message):
    state, _ = chat(OFFER, DOWN, message=message)
    assert state["skipped_asks"] == MAX_SKIPPED_ASKS


def test_backup_a_no_that_answers_a_question_never_parks():
    state, _ = chat("Thanks! Do you own the house?", DOWN, message="no")
    assert state["skipped_asks"] == 0


@pytest.mark.parametrize("message, previous, expected", [
    ("ok", OFFER, "accepts"), ("okay", OFFER, "accepts"), ("fine", OFFER, "accepts"), ("theek hai", OFFER, "accepts"),
    ("ok", "Does that help?", "accepts"),  # a yes/no question that asks for no detail
    ("ok", NOTHING, "closes"), ("ok", CITY_ASK, "skips"),
    ("ok thanks", OFFER, "closes"), ("okay, thank you", OFFER, "closes"), ("fine thanks", OFFER, "closes"),
    ("yes thanks", OFFER, "accepts"), ("sure, thank you", OFFER, "accepts"),
    ("no", "Do you own the house?", "answers"), ("no", OFFER, "declines"),
    ("No, it's rented.", "Do you own the house?", "answers"),
    *[("ok", f"Warranty depends on the equipment. {q}", "closes") for q in OPEN_QUESTIONS],
])
def test_backup_reply_types(message, previous, expected):
    assert reply_type_from_words(message, AIMessage(previous), {}) == expected


@pytest.mark.parametrize("message, kind", [
    ("ok.", "ok"), ("okk", "ok"), ("k", "ok"), ("kk", "ok"), ("okay!", "ok"), ("👍", "ok"), ("theek hai", "ok"),
    ("thanks 🙏", "thanks"), ("thank u", "thanks"), ("thx", "thanks"), ("ok thanks", "thanks"),
    ("yes thanks", "yes"), ("sure, thank you", "yes"), ("ji haan", "yes"),
    ("no thanks", "no"), ("not interested", "no"), ("nahi chahiye", "no"),
    ("ok, what about warranty?", None), ("haha", None), ("Hank", None), ("set up", None),
])
def test_backup_short_replies_match_loosely(message, kind):
    assert short_reply(message) == kind


@pytest.mark.parametrize("previous, kind", [
    (OFFER, "offer"), (CITY_ASK, "detail"), (ROOF_ASK, "detail"), ("Is this for a home or a business?", "detail"),
    ("Does that help?", "yes_no"), (NOTHING, None), *[(q, "open") for q in OPEN_QUESTIONS],
])
def test_last_question_kind_reads_our_own_reply(previous, kind):
    assert last_question_kind(AIMessage(previous)) == kind

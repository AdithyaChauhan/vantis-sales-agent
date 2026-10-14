"""T13d: a ready lead is saved in code when the model didn't save it (its call failed, or it asked again for a detail it
had), details are read in code when the understand step fails, and nothing claims a save that didn't happen. Scripted
models; the real leads service on the Neon test branch (skipped without TEST_DATABASE_URL)."""
import pytest
from sqlalchemy.exc import OperationalError

from app.agent import welcome
from app.agent.leads import LeadOps
from app.leads import service
from app.leads.models import Callback, Lead
from tests.agent_fakes import Scripted, answer, make_graph, run, tool_call, understood
from tests.leads_data import AREAS
from tests.test_agent_leads import count
from tests.test_answer import ReasonLLM

READY = dict(name="Neha Gupta", phone="9876500003", city="Lucknow", monthly_bill_inr="3000", property_type="house")
# The safety net says the city the message names, from the code's own check (KB Section 10)
LUCKNOW_LINE = "Vantis currently covers Lucknow. [KB Section 10: Locations]"


@pytest.fixture
def ops(leads_db):
    return LeadOps(engine=leads_db, areas_fn=lambda: AREAS, reason_llm=ReasonLLM("A ready buyer."))


def test_a_ready_lead_the_model_didnt_save_is_saved_in_code_and_the_reply_says_so(ops):
    """Audit S22 t2: ready to save, but Groq asked for a phone "so we can save your details": no lead."""
    agent = Scripted(answer("Thanks, Neha! Could you share your phone number so we can save your details?"))
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, **READY)), agent, ops=ops)
    state = run(graph, "Neha Gupta, 9876500003, Lucknow, bill 3000, it's a house")
    assert state["lead_id"] and count(ops.engine, Lead) == 1 and ops.calls == ["create"]
    assert state["reply"] == f"Thanks, Neha!\n\n{welcome.SAVED_FALLBACK_TEXT}"  # its question about the phone is gone
    assert state["answered_by"] == "model" and state["lead_created_now"] is True
    assert {"node": "save", "detail": "lead saved (create_lead was not called)",
            "data": {"lead_id": state["lead_id"]}} in state["steps"]


def test_a_ready_lead_is_saved_when_every_model_failed(ops):
    """Render 13 Oct 12:03:32 UTC: name, phone and email, then an 8 s timeout and a Groq 429: no lead was saved."""
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, **READY)), Scripted(RuntimeError("all down")),
                          ops=ops)
    state = run(graph, "Neha Gupta, 9876500003, Lucknow, bill 3000, it's a house")
    assert state["lead_id"] and count(ops.engine, Lead) == 1
    assert state["reply"] == f"{LUCKNOW_LINE}\n\n{welcome.SAVED_FALLBACK_TEXT}"
    assert state["answered_by"] == "knowledge_base"


def test_the_tool_stays_the_normal_path_and_nothing_is_saved_twice(ops):
    agent = Scripted(tool_call("create_lead", {}), answer("Your details are saved."), answer("Anything else?"))
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, **READY), understood(["question"], "q")),
                          agent, ops=ops)
    first = run(graph, "Neha Gupta, 9876500003, Lucknow, bill 3000, it's a house", thread="once")
    run(graph, "What warranty do you give?", thread="once")
    assert first["lead_id"] and not any(s["node"] == "save" for s in first["steps"])
    assert ops.calls == ["create"] and count(ops.engine, Lead) == 1


def test_no_save_in_code_when_they_say_no_or_it_is_a_support_case(ops):
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, declines=True, **READY)),
                          Scripted(answer("No problem.")), ops=ops)
    assert not run(graph, "Neha Gupta, 9876500003, Lucknow, bill 3000. Please don't save anything.").get("lead_id")
    support = understood(["existing_customer_issue"], None, issue="inverter shows an error", **READY)
    graph, _ = make_graph(Scripted(support), Scripted(answer("Sorry to hear that.")), ops=ops)
    assert not run(graph, "My inverter shows an error. Neha Gupta, 9876500003", thread="support").get("lead_id")
    assert count(ops.engine, Lead) == 0 and ops.calls == []


def test_a_callback_time_they_gave_is_logged_with_the_save(ops):
    """Adithya's choice: when finalize saves the lead, a callback time already given is logged too."""
    graph, _ = make_graph(Scripted(understood(["gives_details", "contact_request"], None, callback_time="tomorrow 11am",
                                              **READY)), Scripted(RuntimeError("all down")), ops=ops)
    state = run(graph, "Neha Gupta, 9876500003, Lucknow, bill 3000, house. Call me tomorrow 11am")
    saved = welcome.SAVED_FALLBACK_TEXT.split(" What time")[0]  # the time is known: no question about it
    assert state["reply"] == f"{LUCKNOW_LINE}\n\n{saved} " + welcome.CALLBACK_LOGGED_TEXT.format(time="tomorrow 11am")
    lead = service.get_lead(ops.engine, state["lead_id"])
    assert [c.preferred_time for c in lead.callbacks] == ["tomorrow 11am"] and count(ops.engine, Callback) == 1
    assert lead.intent_level == "ready"  # a request to be called is ready intent
    assert lead.recommended_action.startswith("Call tomorrow 11am, as requested")


def test_a_database_error_in_the_code_save_saves_nothing_and_claims_nothing(ops, monkeypatch):
    def down(*args, **kwargs):
        raise OperationalError("INSERT INTO leads", {}, Exception("down"))

    monkeypatch.setattr(service, "create_lead", down)
    agent = Scripted(answer("Thanks! I've saved your details."))
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, **READY)), agent, ops=ops)
    state = run(graph, "Neha Gupta, 9876500003, Lucknow, bill 3000, it's a house")
    assert not state.get("lead_id") and "saved" not in state["reply"].lower()


def test_details_read_in_code_when_understand_fails_still_save_the_lead(ops):
    """The city and need came earlier; the turn with name, phone and email had no understand model at all."""
    understand_llm = Scripted(understood(["buying_interest", "gives_details"], None, city="Lucknow",
                                         monthly_bill_inr="3500", property_type="house"),
                              RuntimeError("main and fast down"))
    agent = Scripted(answer("Lucknow is covered [KB Section 10: Locations]. Could you share your name and phone?"),
                     RuntimeError("all down"))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    run(graph, "mera ghar Lucknow mein hai, bill 3500", thread="raw")
    state = run(graph, "main Neha Gupta hu, mera number 9876500003 hai, email neha@example.com", thread="raw")
    understand_step = state["steps"][0]
    assert understand_step["detail"] == "failed, raw message used"
    assert understand_step["data"]["from_text"] == ["email", "name", "phone"]
    lead = service.get_lead(ops.engine, state["lead_id"])
    assert (lead.name, lead.phone, lead.email, lead.city) == ("Neha Gupta", "+919876500003", "neha@example.com",
                                                               "Lucknow")
    assert state["reply"] == welcome.SAVED_FALLBACK_TEXT


def test_home_or_business_counts_as_the_need(ops):
    """Audit S19: "for my home" with name, phone and city saved nothing; the type is now the need."""
    details = dict(name="Amit", phone="9876500004", city="Kanpur", customer_type="residential")
    graph, _ = make_graph(Scripted(understood(["gives_details"], None, **details)), Scripted(answer("Thanks!")),
                          ops=ops)
    assert run(graph, "Amit, 9876500004, Kanpur, it's for my home")["lead_id"]


# ---------- T13d round 2: exploring and timelines read in code; the action for a call asked for without a time ----------

def test_just_exploring_ends_cold_when_understand_fails_every_turn(ops):
    """The live check's D7 on 13 Oct: Google timed out on every understand call and the lead was saved WARM 46."""
    graph, _ = make_graph(Scripted(RuntimeError("main and fast down")), Scripted(RuntimeError("all down")), ops=ops)
    run(graph, "Just exploring solar for now, maybe next year", thread="d7")
    state = run(graph, "I'm Vikram Rao, 9876500004, house in Kanpur, bill around 3000", thread="d7")
    lead = service.get_lead(ops.engine, state["lead_id"])
    assert (lead.name, lead.intent_level, lead.timeline_bucket) == ("Vikram Rao", "exploring", "gt_6m")
    assert (lead.lead_score, lead.lead_status, [c["rule"] for c in lead.score_breakdown["caps"]]) == (
        44, "COLD", ["exploring"])


def test_a_call_asked_for_without_a_time_is_in_the_action(ops):
    understand_llm = Scripted(understood(["contact_request"], None),
                              understood(["gives_details"], None, **READY),
                              understood(["question"], "warranty"))
    agent = Scripted(answer("Sure, could you share your name and phone?"), tool_call("create_lead", {}),
                     answer("Your details are saved."), answer("Warranty depends on the equipment."))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    run(graph, "Can someone call me?", thread="call")
    state = run(graph, "Neha Gupta, 9876500003, Lucknow, bill 3000, it's a house", thread="call")
    lead = service.get_lead(ops.engine, state["lead_id"])
    assert lead.recommended_action.startswith("Call the customer within 24 hours; they asked for a call.")
    run(graph, "What warranty do you give?", thread="call")  # a later update keeps it
    assert service.get_lead(ops.engine, state["lead_id"]).recommended_action.startswith("Call the customer")


def test_a_call_asked_for_after_the_save_updates_the_action(ops):
    """Nothing else changes on that turn (the intent is already ready), so only the action is written again."""
    understand_llm = Scripted(understood(["gives_details"], None, intent_level="ready", **READY),
                              understood(["contact_request"], None))
    agent = Scripted(tool_call("create_lead", {}), answer("Your details are saved."), answer("Sure, the team can call."))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    first = run(graph, "Neha Gupta, 9876500003, Lucknow, bill 3000, it's a house", thread="later")
    assert not service.get_lead(ops.engine, first["lead_id"]).recommended_action.startswith("Call the customer")
    state = run(graph, "Please have someone call me", thread="later")
    assert service.get_lead(ops.engine, first["lead_id"]).recommended_action.startswith(
        "Call the customer within 24 hours; they asked for a call.")
    assert any(s["node"] == "sync" and s["detail"] == "the action now says to call them" for s in state["steps"])

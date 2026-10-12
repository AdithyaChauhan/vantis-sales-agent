"""Lead tools in the agent: scripted models, the real T7 leads service on the Neon test branch."""
import json
import threading
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent import welcome
from app.agent.leads import (
    ASKED_WHAT_LINE, INTEREST_LINE, SAVED_INFO_LINE, LeadOps, card_fields, confirmed_city, details_named,
    fix_callback_claims, is_clock_time, merge_callback_time, remove_false_claims, ruled_out_details,
    short_answer_details, status_block,
)
from app.leads import service
from app.leads.models import Callback, Lead, SupportRequest
from tests.agent_fakes import Scripted, answer, make_graph, run, tool_call, understood
from tests.leads_data import AREAS
from tests.test_answer import FakeLLM, ReasonLLM

REASON = "Ready buyer with a clear 5 kW requirement in a listed city."
FORBIDDEN_IN_PROMPT = ("lead_score", "lead_status", "HOT", "WARM", "COLD", REASON, "score")


@pytest.fixture
def ops(leads_db):
    return LeadOps(engine=leads_db, areas_fn=lambda: AREAS, reason_llm=ReasonLLM(REASON))


def count(engine, model) -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(model))


def dynamic_inputs(agent: Scripted) -> str:
    """What code adds to the model's input each turn: the lead progress blocks and the tool results
    (the static system prompt itself names the forbidden words in its own rule)."""
    parts = []
    for call in agent.calls:
        for m in call:
            content = str(m.content)
            if isinstance(m, ToolMessage):
                parts.append(content)
            elif "<lead_progress>" in content:
                parts.append(content.split("<lead_progress>")[1].split("</lead_progress>")[0])
    return "\n".join(parts)


def lead_status_text(agent: Scripted, call: int = -1) -> str:
    content = str(agent.calls[call][-1].content) if not isinstance(agent.calls[call][-1], ToolMessage) else ""
    if "<lead_progress>" not in content:
        content = next(str(m.content) for m in reversed(agent.calls[call]) if "<lead_progress>" in str(m.content))
    return content.split("<lead_progress>")[1].split("</lead_progress>")[0]


DEMO_DETAILS = dict(
    city="Lucknow", system_size_kw="5 kW", monthly_bill_inr="₹4,000", timeline="next month",
    timeline_bucket="lt_1m", intent_level="ready", property_type="house",
)


def demo_graph(ops):
    understand_llm = Scripted(
        understood(["question"], "Product & Service Catalog offerings"),
        understood(["buying_interest", "gives_details"], "Pricing 5 kW residential", **DEMO_DETAILS),
        understood(["gives_details"], None, name="Rahul Sharma", phone="98765 43210",
                   email="rahul.sharma@example.com"),
        understood(["contact_request", "reply_to_assistant"], None, callback_time="tomorrow evening"),
    )
    agent = Scripted(
        answer("We offer Plan A and Plan B [KB Section 8: Catalog]."),
        tool_call("check_service_area", {"city": "Lucknow"}, "c1"),
        answer("Lucknow is listed [KB Section 10: Locations]. Could you share your name and phone number?"),
        tool_call("create_lead", {}, "c2"),
        answer("Thanks, I've saved your details. Would you like a callback?"),
        tool_call("schedule_callback", {}, "c3"),
        answer("I've logged a callback request for tomorrow evening; our team will confirm."),
    )
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    return graph, agent


def test_demo_flow_creates_a_hot_lead_and_a_requested_callback(ops):
    graph, agent = demo_graph(ops)
    run(graph, "What systems do you offer for homes?", thread="demo")
    second = run(graph, "5 kW for my house in Lucknow, bill about 4000, next month", thread="demo")
    draft = second["lead_summary"]
    assert draft["missing_fields"] == ["name", "phone or email"]
    assert (draft["name"], draft["city"], draft["property_type"], draft["customer_type"]) == (
        None, "Lucknow", "house", "residential")
    assert {"city", "monthly_bill_inr", "system_size_kw", "timeline"} <= set(draft["known_fields"])
    assert "4000" not in json.dumps(draft) and "4,000" not in json.dumps(draft)  # field names, never amounts
    assert "Still needed before it can be saved: name, phone or email" in lead_status_text(agent)
    third = run(graph, "Rahul Sharma, 98765 43210, rahul.sharma@example.com", thread="demo")
    assert [s.get("name") for s in third["steps"] if s["node"] == "tool"] == ["create_lead"]
    assert third["lead_id"] and third["lead_summary"]["status"] == "HOT"
    saved = third["lead_summary"]
    assert (saved["name"], saved["city"], saved["customer_type"]) == ("Rahul Sharma", "Lucknow", "residential")
    assert {"phone", "email", "monthly_bill_inr"} <= set(saved["known_fields"])
    for value in ("98765", "43210", "rahul.sharma@example.com", "4000"):  # names only, no contact or money
        assert value not in json.dumps(saved), value
    fourth = run(graph, "Yes, call me tomorrow evening", thread="demo")
    assert fourth["callback_status"] == "requested"
    assert fourth["lead_summary"]["callback_status"] == "requested"

    lead = service.get_lead(ops.engine, third["lead_id"])
    assert (lead.lead_score, lead.lead_status, lead.location_status) == (97, "HOT", "listed")
    # The callback time replaces "within 1 hour" (T12a)
    assert lead.reason == REASON
    assert lead.recommended_action == "Call tomorrow evening, as requested, and offer a site assessment."
    assert [c.preferred_time for c in lead.callbacks] == ["tomorrow evening"]
    assert lead.callbacks[0].status == "requested"
    assert ops.calls == ["create", "callback"]  # each write once; no extra sync
    assert count(ops.engine, Lead) == 1
    # The model never saw a score, a status or the reason
    seen = dynamic_inputs(agent)
    assert "Lead saved: yes" in seen and '"logged": true' in seen  # the checks below look at real content
    for word in FORBIDDEN_IN_PROMPT:
        assert word not in seen, word


def test_incomplete_lead_is_not_saved(ops):
    understand_llm = Scripted(understood(["gives_details"], None, name="Rahul Sharma", city="Lucknow"))
    agent = Scripted(tool_call("create_lead", {}), answer("Could you share a phone number or email?"))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    state = run(graph, "I'm Rahul from Lucknow")
    tool_result = agent.calls[1][-1]
    assert '"saved": false' in tool_result.content and "phone or email" in tool_result.content
    assert state.get("lead_id") is None and count(ops.engine, Lead) == 0
    assert state["lead_summary"]["missing_fields"][0] == "requirement, monthly bill, usage, system size or property type"


def test_contact_request_skips_the_search_and_asks_for_name_and_phone(ops):
    understand_llm = Scripted(
        understood(["question"], "service area Delhi", asked_city="Delhi"),
        understood(["contact_request", "reply_to_assistant"], "Product & Service Catalog"),
    )
    agent = Scripted(answer("Delhi is in our listed service areas. Is the installation for a home or a business?"),
                     answer("Sure, could you share your name and phone number?"))
    graph, search = make_graph(understand_llm, agent, ops=ops)
    run(graph, "Do you install in Delhi?")
    state = run(graph, "please have someone call me")
    assert search.queries == ["service area Delhi"]  # no KB search for the contact request
    block = lead_status_text(agent)
    assert "Still needed before it can be saved: name, phone or email" in block
    assert "asked to be contacted" in block
    assert state["contact_request"] is True


def test_invalid_phone_is_asked_again_and_never_saved(ops):
    understand_llm = Scripted(understood(["gives_details"], None, name="Rahul Sharma", phone="12345", city="Lucknow",
                                         system_size_kw=5))
    agent = Scripted(answer("That number seems short; could you check it?"))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    state = run(graph, "Rahul, 12345")
    block = lead_status_text(agent)
    assert "Looked wrong, ask again: phone: That doesn't look like a 10-digit Indian mobile number" in block
    assert "12345" not in block  # the message, never the value
    assert "phone" not in state["lead"] and count(ops.engine, Lead) == 0


def test_ready_to_save_line(ops):
    understand_llm = Scripted(understood(["gives_details"], None, name="Asha Rao", email="asha@example.com",
                                         city="Kanpur", property_type="house"))
    agent = Scripted(answer("ok"))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    run(graph, "details")
    assert "Ready to save: call create_lead now" in lead_status_text(agent)


@pytest.mark.parametrize(
    "reply, out, expected",
    [
        ("yes", {"intents": ["reply_to_assistant"], "details": {}}, "Delhi"),
        ("Rahul Sharma, 9876543210", {"intents": ["gives_details"], "details": {"name": "Rahul Sharma"}}, "Delhi"),
        ("no", {"intents": ["reply_to_assistant"], "details": {}, "declines": True}, None),
        ("it's in Noida", {"intents": ["gives_details"], "details": {"city": "Noida"}}, None),
        ("tell me a joke", {"intents": ["off_topic"], "details": {}}, None),
    ],
)
def test_city_confirmation_rule(reply, out, expected):
    question = AIMessage("Yes, Delhi is in our listed areas. Is this for the property in Delhi?")
    assert confirmed_city(["Delhi"], question, out) == expected


def test_city_not_confirmed_when_the_question_did_not_name_it():
    question = AIMessage("Delhi is listed. Is the installation for a home or a business?")
    assert confirmed_city(["Delhi"], question, {"intents": ["reply_to_assistant"], "details": {}}) is None


def test_city_confirmation_through_the_graph(ops):
    understand_llm = Scripted(
        understood(["question"], "service area Delhi", asked_city="Delhi"),
        understood(["gives_details"], None, name="Rahul Sharma", phone="9876543210"),
    )
    agent = Scripted(answer("Delhi is listed. Could I have your name and number for the property in Delhi?"),
                     answer("Thanks."))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    first = run(graph, "Do you install in Delhi?")
    assert first["asked_cities"] == ["Delhi"] and "city" not in first["lead"]
    assert "City asked about, not confirmed as theirs: Delhi" in lead_status_text(agent, 0)
    second = run(graph, "Rahul Sharma, 9876543210")
    assert second["lead"]["city"] == "Delhi" and second["asked_cities"] == []
    assert second["steps"][0]["data"]["city_confirmed"] is True


def test_information_only_turn_does_not_ask_for_details():
    info = {"lead": {"intent_level": "exploring"}, "intents": ["question"]}  # intent_level is a guess, not a detail
    block = status_block(info)
    assert "do not ask for their details" in block and "Still needed" not in block


@pytest.mark.parametrize("state", [
    {"lead": {}, "intents": ["question", "buying_interest"]},
    {"lead": {"system_size_kw": 5.0}, "intents": ["question"]},
    {"lead": {}, "intents": ["question"], "asked_cities": ["Karimnagar"]},
])
def test_interest_brings_back_the_missing_details(state):
    block = status_block(state)
    assert "Still needed before it can be saved" in block and "do not ask for their details" not in block


def test_card_fields_send_names_and_never_contact_or_money_values():
    values = {"name": "Rahul Sharma", "city": "Lucknow", "phone": "+919876543210", "email": "r@example.com",
              "monthly_bill_inr": 4000, "property_type": "house", "customer_type": "residential", "timeline": None}
    card = card_fields(values)
    assert (card["name"], card["city"], card["property_type"], card["customer_type"]) == (
        "Rahul Sharma", "Lucknow", "house", "residential")
    assert card["known_fields"] == ["city", "customer_type", "email", "monthly_bill_inr", "name", "phone",
                                    "property_type"]  # a null field is not known
    for value in ("+919876543210", "r@example.com", 4000):
        assert value not in card.values() and str(value) not in str(card), value


def test_two_asked_cities_ask_which():
    state = {"lead": {}, "asked_cities": ["Delhi", "Kanpur"]}
    assert "Ask which one the property is in" in status_block(state)


def test_inverter_issue_becomes_a_support_request_not_a_lead(ops):
    understand_llm = Scripted(
        understood(["existing_customer_issue"], "Reference Support Scenarios inverter issue",
                   issue="inverter shows a red light", project_details="3 kW, installed 2024 in Kanpur"),
        understood(["gives_details"], None, name="Anil Kumar", phone="9876511111"),
    )
    agent = Scripted(
        answer("Sorry about that. Could you share your name and phone number?"),
        tool_call("create_lead", {}, "c1"),  # a wrong call: the guard refuses it
        tool_call("create_support_request", {}, "c2"),
        answer("I've saved your support request."),
    )
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    first = run(graph, "my inverter shows a red light")
    assert "Still needed for the support request: name, phone or email" in lead_status_text(agent, 0)
    assert first["lead_summary"] is None
    second = run(graph, "Anil Kumar, 9876511111")
    refused = next(m for m in agent.calls[2] if isinstance(m, ToolMessage))
    assert "use create_support_request" in refused.content
    assert second["support_request_id"] and count(ops.engine, SupportRequest) == 1
    assert count(ops.engine, Lead) == 0 and ops.calls == ["support"]


def test_callback_needs_a_lead_and_a_time(ops):
    understand_llm = Scripted(understood(["contact_request"], None, callback_time="after 6"))
    agent = Scripted(tool_call("schedule_callback", {}), answer("Could I have your name and phone first?"))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    run(graph, "call me after 6")
    result = agent.calls[1][-1].content
    assert '"logged": false' in result and "save the lead first" in result
    assert count(ops.engine, Callback) == 0


def test_new_details_after_saving_are_written_once(ops):
    understand_llm = Scripted(
        understood(["gives_details"], None, name="Rahul Sharma", phone="9876543210", city="Lucknow", system_size_kw=5),
        understood(["gives_details"], None, email="rahul.sharma@example.com"),  # finalize syncs it
        understood(["gives_details"], None, ownership="own"),  # the update_lead tool saves it
    )
    agent = Scripted(
        tool_call("create_lead", {}, "c1"), answer("Saved."),
        answer("Thanks, noted."),
        tool_call("update_lead", {}, "c2"), answer("Updated."),
    )
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    run(graph, "details")
    second = run(graph, "my email")
    assert [s["node"] for s in second["steps"]][-1] == "sync"
    run(graph, "I own it")
    assert ops.calls == ["create", "sync", "update"]  # never a second write for the same details
    lead = service.get_lead(ops.engine, second["lead_id"])
    assert (lead.email, lead.ownership) == ("rahul.sharma@example.com", "own")


def test_two_concurrent_creates_for_one_session_give_one_lead(ops):
    raw = {"name": "Rahul Sharma", "phone": "9876543210", "city": "Lucknow", "system_size_kw": 5}
    errors = []

    def create():
        try:
            service.create_lead(ops.engine, raw, session_id="race-1", llm=FakeLLM(REASON), areas=AREAS)
        except Exception as exc:  # pragma: no cover - would fail the test below
            errors.append(exc)

    threads = [threading.Thread(target=create) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and count(ops.engine, Lead) == 1


def test_lead_tool_results_never_hold_a_score(ops):
    understand_llm = Scripted(understood(["gives_details"], None, name="Rahul Sharma", phone="9876543210",
                                         city="Lucknow", system_size_kw=5))
    agent = Scripted(tool_call("create_lead", {}), answer("Saved."))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    state = run(graph, "details")
    result = agent.calls[1][-1].content
    assert set(__import__("json").loads(result)) == {"saved", "lead_id", "fields_saved"}
    assert state["lead_summary"]["score"] is not None  # the API decides whether to show it


def test_after_saving_the_block_offers_a_callback_first():
    saved = {"lead": {"name": "Rahul Sharma"}, "lead_id": 3, "lead_created_now": True}
    block = status_block(saved)
    assert "offer a callback" in block and "Do not ask for more details in this reply" in block
    later = status_block({**saved, "lead_created_now": False, "callback_status": "requested"})
    assert "offer a callback" not in later


def test_reason_is_written_after_the_reply(ops, client):
    from app.agent.graph import get_agent
    from app.main import app

    ops.reason_now = False
    understand_llm = Scripted(understood(["gives_details"], None, name="Rahul Sharma", phone="9876543210",
                                         city="Lucknow", system_size_kw=5))
    agent = Scripted(tool_call("create_lead", {}), answer("Saved; the sales team can follow up."))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    app.dependency_overrides[get_agent] = lambda: graph
    body = client.post("/api/chat", json={"message": "details", "session_id": str(uuid.uuid4())}).json()
    lead = service.get_lead(ops.engine, body["lead"]["id"])
    # TestClient runs the background task before returning, as the server does after the response.
    # A WARM lead's reason ends with what keeps it from HOT (T12a).
    assert lead.reason_source == "llm" and lead.reason.startswith(REASON.rstrip("."))
    assert "not HOT yet:" in lead.reason


def test_refresh_reason_skips_a_lead_whose_score_changed(ops):
    raw = {"name": "Rahul Sharma", "phone": "9876543210", "city": "Lucknow", "system_size_kw": 5}
    lead = service.create_lead(ops.engine, raw, areas=AREAS, reason_now=False).lead
    assert lead.reason_source == "template"

    class ChangesScoreFirst(ReasonLLM):
        def invoke(self, messages):
            service.update_lead(ops.engine, lead.id, {"email": "rahul.sharma@example.com"}, reason_now=False)
            return super().invoke(messages)

    assert service.refresh_reason(ops.engine, lead.id, ChangesScoreFirst(REASON)) is False
    assert service.get_lead(ops.engine, lead.id).reason_source == "template"


# ---------- T12a: the problems from Adithya's test ----------

def test_the_2000_turn_never_claims_a_save_before_a_lead_exists(ops):
    # Problem 1: after "2000" the bot said "Your details are saved" with no name or phone yet
    understand_llm = Scripted(understood(["gives_details"], None, city="Noida", monthly_bill_inr="2000"))
    agent = Scripted(answer("Thanks, a bill of 2,000 a month helps. Your details are saved and our team "
                            "will be in touch soon."))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    state = run(graph, "2000")
    assert state["reply"] == ("Thanks, a bill of 2,000 a month helps. Could you share your name and phone number? "
                              "Then I can pass your details to the sales team.")
    assert any(s["node"] == "guard" for s in state["steps"])
    assert state.get("lead_id") is None and count(ops.engine, Lead) == 0
    block = lead_status_text(agent)
    assert "never say the details are saved, recorded or passed to the team" in block


@pytest.mark.parametrize("reply", [
    "Your details are saved.", "I've saved your details.", "I have recorded your bill.",
    "Your details have been passed to our sales team.", "I've logged a callback request for 4 PM.",
    "Saved! The team can follow up.",
])
def test_save_claims_are_removed_while_no_lead_exists(reply):
    state = {"lead": {"city": "Noida", "monthly_bill_inr": 2000}}
    text, changed = remove_false_claims(f"Thanks. {reply}", state)
    assert changed
    assert text.startswith("Thanks.") and text.endswith(welcome.STILL_NEEDED_ASK.format(items="name and phone number"))


@pytest.mark.parametrize("reply", [
    "Once you share your phone number, I can pass your details to the team.",
    "Your details will be saved when you share a phone number.",
    "I've shared the warranty terms above.",
    "Thanks, noted.",
])
def test_future_or_unrelated_wording_is_kept(reply):
    assert remove_false_claims(reply, {"lead": {"city": "Noida"}}) == (reply, False)


def test_claims_are_kept_when_a_lead_or_support_request_exists_and_a_question_stays():
    assert remove_false_claims("Your details are saved.", {"lead_id": 4}) == ("Your details are saved.", False)
    assert remove_false_claims("I've logged your issue.", {"support_request_id": 2})[1] is False
    # A question is left, so nothing is appended (one question per reply)
    text, changed = remove_false_claims("Your details are saved. Could you share your name?", {"lead": {}})
    assert (text, changed) == ("Could you share your name?", True)
    # Everything needed is there but create_lead was never called: ask before passing anything on
    ready = {"lead": {"name": "Asha Rao", "phone": "+919876543210", "city": "Noida", "monthly_bill_inr": 2000}}
    assert remove_false_claims("All saved.", ready) == (welcome.READY_TO_PASS_ASK, True)


@pytest.mark.parametrize("question, message, expected", [
    ("Thanks! What is your average monthly electricity bill?", "2000", {"monthly_bill_inr": "2000"}),
    ("What is your monthly bill, roughly?", "₹3,500", {"monthly_bill_inr": "₹3,500"}),
    ("Does the property have usable roof space for the panels?", "yes", {"rooftop_available": True}),
    ("Does the property have usable roof space for the panels?", "nope", {"rooftop_available": False}),
    ("What size system or monthly bill do you have in mind?", "5", {}),  # two details asked: not guessed
    ("Do you own the property, and is there roof space?", "yes", {}),
    ("Would you like a callback?", "2000", {}),
    ("What is your monthly electricity bill?", "about 2000 i think", {}),  # not bare: the extractor's job
])
def test_short_answers_fill_the_one_detail_asked_about(question, message, expected):
    assert short_answer_details(AIMessage(question), message, {}) == expected


def test_a_short_answer_the_extractor_missed_is_saved_and_not_asked_again(ops):
    understand_llm = Scripted(
        understood(["gives_details"], None, city="Noida"),
        understood(["reply_to_assistant"], None),  # the extractor missed the bare number
    )
    agent = Scripted(answer("Vantis covers Noida. What is your average monthly electricity bill?"),
                     answer("Thanks. Could I have your name and phone number?"))
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    run(graph, "Noida")
    state = run(graph, "2000")
    assert state["lead"]["monthly_bill_inr"] == 2000
    assert state["steps"][0]["data"]["short_answer"] == ["monthly_bill_inr"]
    assert "monthly_bill_inr" in lead_status_text(agent).split("Details known: ")[1].split(".")[0]


@pytest.mark.parametrize("old, new, merged", [
    ("4pm", "day after tomorrow", "day after tomorrow, 4pm"),
    ("tomorrow evening", "6 pm", "tomorrow, 6 pm"),
    ("tomorrow 4pm", "Friday", "Friday, 4pm"),
    ("4pm", "5pm", "5pm"),
    ("tomorrow", "next week", "next week"),
    ("Monday 10 am", "Tuesday 11 am", "Tuesday 11 am"),
    (None, "4pm", "4pm"),
    ("4pm", None, "4pm"),
])
def test_callback_time_merge(old, new, merged):
    assert merge_callback_time(old, new) == merged


def saved_lead_graph(ops, *understandings, answers):
    """A lead saved on the first turn, then the given turns."""
    details = dict(name="Rahul Sharma", phone="9876543210", city="Lucknow", system_size_kw=5)
    understand_llm = Scripted(understood(["gives_details"], None, **details), *understandings)
    agent = Scripted(tool_call("create_lead", {}, "c0"), answer("Saved; the team can follow up. When suits you?"),
                     *answers)
    graph, _ = make_graph(understand_llm, agent, ops=ops)
    run(graph, "details")
    return graph, agent


def test_a_new_callback_time_updates_the_one_request(ops):
    # Problem 5: "4pm", then "get back to me day after tomorrow" gave two callback requests
    graph, agent = saved_lead_graph(
        ops,
        understood(["contact_request"], None, callback_time="4pm"),
        understood(["contact_request"], None, callback_time="day after tomorrow"),
        answers=[tool_call("schedule_callback", {}, "c1"), answer("I have logged a callback request for 4pm."),
                 tool_call("schedule_callback", {}, "c2"),
                 answer("I have updated your callback request to day after tomorrow, 4pm.")],
    )
    first = run(graph, "4pm")
    assert first["callback_logged"] == "4pm" and first["callback_now"] == "requested"
    assert "The customer gave a preferred callback time: call schedule_callback." in lead_status_text(agent, 2)
    second = run(graph, "get back to me day after tomorrow")
    assert second["callback_time"] == "day after tomorrow, 4pm"
    assert ("The customer gave a new callback time: call schedule_callback to update the request."
            in lead_status_text(agent, 4))
    result = json.loads(agent.calls[5][-1].content)
    assert result["updated"] is True and result["preferred_time"] == "day after tomorrow, 4pm"
    assert [s["detail"] for s in second["steps"] if s.get("name") == "schedule_callback"] == ["updated"]
    lead = service.get_lead(ops.engine, second["lead_id"])
    assert [(c.preferred_time, c.status) for c in lead.callbacks] == [("day after tomorrow, 4pm", "requested")]
    assert lead.recommended_action.startswith("Call day after tomorrow, 4pm, as requested.")
    assert count(ops.engine, Callback) == 1


SAVED = {"lead": {"name": "Asha Rao", "phone": "+919876543210", "city": "Noida", "monthly_bill_inr": 2000,
                  "requirement": "solar for my office", "property_type": "office"}, "lead_id": 7}


def test_a_saved_lead_lists_what_would_still_add_points_most_useful_first():
    # Problem 2: "what else do you need?" got "nothing more" while timeline, rooftop, ownership and email were missing
    block = status_block({**SAVED, "intents": ["asks_what_is_needed"]})
    assert "Would still add points, most useful first: timeline, rooftop availability, ownership, email." in block
    assert "Never ask again for anything under Details known." in block
    assert ASKED_WHAT_LINE.format(detail="timeline") in block  # the most useful one, chosen in code
    for known in ("monthly bill", "phone number", "city"):  # given already: never in the list
        assert known not in block.split("most useful first: ")[1].split(".")[0]


@pytest.mark.parametrize("message, detail", [
    # T12a replay, turn 9: "apart from timing" still got a timeline question
    ("is there any other info you need from me apart from the timing?", "rooftop availability"),
    # turn 10: the detail the customer asks about comes first
    ("do you really need the email?", "email"),
    ("what else?", "timeline"),
])
def test_the_detail_to_ask_for_is_chosen_in_code(message, detail):
    state = {**SAVED, "intents": ["asks_what_is_needed"], "messages": [HumanMessage(message)],
             "ruled_out": ruled_out_details(message)}
    assert ASKED_WHAT_LINE.format(detail=detail) in status_block(state)


def test_a_ruled_out_detail_is_never_offered_again():
    state = {**SAVED, "intents": ["gives_details"], "new_details": ["rooftop_available"], "ruled_out": ["timeline"]}
    block = status_block(state)
    assert "most useful first: rooftop availability, ownership, email." in block
    assert INTEREST_LINE.format(detail="rooftop availability") in block
    assert ruled_out_details("anything else apart from timing?") == ["timeline"]
    assert details_named("is rooftop availability info not needed?") == ["rooftop availability"]


@pytest.mark.parametrize("turn, line", [
    ({"intents": ["gives_details"], "new_details": ["rooftop_available"]}, INTEREST_LINE.format(detail="timeline")),
    ({"intents": ["buying_interest"]}, INTEREST_LINE.format(detail="timeline")),
    ({"intents": ["question"]}, SAVED_INFO_LINE),  # only information: don't push for details
    ({"intents": ["reply_to_assistant", "off_topic"], "declines": True}, SAVED_INFO_LINE),
])
def test_a_saved_lead_asks_for_one_detail_only_when_the_customer_shows_interest(turn, line):
    block = status_block({**SAVED, **turn})
    assert line in block
    others = {INTEREST_LINE.format(detail="timeline"), SAVED_INFO_LINE, ASKED_WHAT_LINE.format(detail="timeline")}
    assert not any(other in block for other in others - {line})


@pytest.mark.parametrize("text, clock", [
    ("4pm", True), ("around 6 pm", True), ("evening", True), ("16:30", True),
    ("next month", False), ("tomorrow 4pm", False), ("in 2 weeks", False), ("as soon as possible", False),
])
def test_a_time_of_day_alone_is_never_a_timeline(text, clock):
    assert is_clock_time(text) is clock


def test_the_replays_callback_turns_end_with_one_request_and_true_sentences(ops):
    # T12a replay, turns 11-13: "yes" (roof), "4pm" (meant for the callback; the extractor made it the timeline),
    # then "get back to me day after tomorrow". The reply claimed a callback the tool had refused, then "updated" it.
    graph, agent = saved_lead_graph(
        ops,
        understood(["gives_details", "reply_to_assistant"], None, timeline="4pm", timeline_bucket="lt_1m"),
        understood(["contact_request"], None, callback_time="day after tomorrow"),
        answers=[tool_call("schedule_callback", {}, "c1"),
                 answer("I have logged a callback request for 4 PM, and the sales team will confirm it."),
                 tool_call("schedule_callback", {}, "c2"),
                 answer("I have updated your callback request to the day after tomorrow, and the team will confirm it.")],
    )
    first = run(graph, "4pm")
    assert "timeline" not in first["lead"] and first["callback_time"] == "4pm"  # a time to call, not a timeline
    assert first["reply"] == welcome.CALLBACK_LOGGED_TEXT.format(time="4pm")
    second = run(graph, "get back to me day after tomorrow")
    assert second["reply"] == welcome.CALLBACK_UPDATED_TEXT.format(time="day after tomorrow, 4pm")
    lead = service.get_lead(ops.engine, second["lead_id"])
    assert [(c.preferred_time, c.status) for c in lead.callbacks] == [("day after tomorrow, 4pm", "requested")]
    assert lead.timeline is None


def test_a_bare_time_of_day_is_a_callback_time_even_when_the_extractor_misses_it(ops):
    # T12a replay (final run), turn 12: after an ownership question "4pm" came back with no details at all
    graph, agent = saved_lead_graph(
        ops,
        understood(["reply_to_assistant"], None),
        understood(["contact_request"], None, callback_time="day after tomorrow"),
        answers=[tool_call("schedule_callback", {}, "c1"), answer("Logged your callback request for 4pm."),
                 tool_call("schedule_callback", {}, "c2"),
                 answer("I have updated your callback request to the day after tomorrow.")],
    )
    first = run(graph, "4pm")
    assert first["callback_time"] == "4pm" and first["callback_now"] == "requested"
    assert first["reply"] == welcome.CALLBACK_LOGGED_TEXT.format(time="4pm")
    second = run(graph, "get back to me day after tomorrow")
    assert second["reply"] == welcome.CALLBACK_UPDATED_TEXT.format(time="day after tomorrow, 4pm")
    lead = service.get_lead(ops.engine, second["lead_id"])
    assert [(c.preferred_time, c.status) for c in lead.callbacks] == [("day after tomorrow, 4pm", "requested")]


@pytest.mark.parametrize("reply, state, expected", [
    # No callback exists: the claim goes, and the reply asks for a time
    ("I have logged a callback request for 4 PM.", {"lead_id": 3}, welcome.CALLBACK_TIME_ASK),
    # An earlier request: "booked" or "updated" becomes "is logged"; a plain restatement stays
    ("Your callback is booked for 4pm.", {"lead_id": 3, "callback_status": "requested", "callback_logged": "4pm"},
     welcome.CALLBACK_EXISTS_TEXT.format(time="4pm")),
    ("Your callback request for 4pm is logged.", {"lead_id": 3, "callback_status": "requested",
                                                  "callback_logged": "4pm"}, "Your callback request for 4pm is logged."),
    # Written this turn as a new request, but the reply says "updated": the true sentence instead, once
    ("I've updated your callback request to Friday. Your call back is set up.",
     {"lead_id": 3, "callback_status": "requested", "callback_now": "requested", "callback_logged": "Friday"},
     welcome.CALLBACK_LOGGED_TEXT.format(time="Friday")),
    # A question about a callback is never a claim
    ("Would you like a callback?", {"lead_id": 3}, "Would you like a callback?"),
])
def test_callback_sentences_say_what_schedule_callback_did(reply, state, expected):
    assert fix_callback_claims(reply, state)[0] == expected


def test_asking_what_is_needed_skips_the_kb_search(ops):
    # Problem 3: "are you sure? rooftop availability info is not needed?" got a KB quote
    graph, agent = saved_lead_graph(
        ops, understood(["asks_what_is_needed"], "Residential requirements rooftop"),
        answers=[answer("It does help: does the property have usable roof space for the panels?")],
    )
    state = run(graph, "is roof space info needed?")
    assert not any(s["node"] == "retrieve" for s in state["steps"])  # no KB context, so nothing to quote
    assert ASKED_WHAT_LINE.format(detail="rooftop availability") in lead_status_text(agent)

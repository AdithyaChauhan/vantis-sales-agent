"""Lead tools in the agent: scripted models, the real T7 leads service on the Neon test branch."""
import threading
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.agent.leads import LeadOps, confirmed_city, status_block
from app.core.db import init_db
from app.leads import service
from app.leads.models import Callback, Lead, SupportRequest
from tests.agent_fakes import Scripted, answer, make_graph, run, tool_call, understood
from tests.leads_data import AREAS
from tests.test_answer import FakeLLM

REASON = "Ready buyer with a clear 5 kW requirement in a listed city."
FORBIDDEN_IN_PROMPT = ("lead_score", "lead_status", "HOT", "WARM", "COLD", REASON, "score")


@pytest.fixture
def ops(test_engine):
    init_db(test_engine)
    with test_engine.begin() as conn:
        conn.execute(text("TRUNCATE leads, callbacks, support_requests RESTART IDENTITY CASCADE"))
    return LeadOps(engine=test_engine, areas_fn=lambda: AREAS, reason_llm=FakeLLM(REASON))


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
    assert second["lead_summary"]["missing_fields"] == ["name", "phone or email"]
    assert "Still needed before it can be saved: name, phone or email" in lead_status_text(agent)
    third = run(graph, "Rahul Sharma, 98765 43210, rahul.sharma@example.com", thread="demo")
    assert [s.get("name") for s in third["steps"] if s["node"] == "tool"] == ["create_lead"]
    assert third["lead_id"] and third["lead_summary"]["status"] == "HOT"
    fourth = run(graph, "Yes, call me tomorrow evening", thread="demo")
    assert fourth["callback_status"] == "requested"
    assert fourth["lead_summary"]["callback_status"] == "requested"

    lead = service.get_lead(ops.engine, third["lead_id"])
    assert (lead.lead_score, lead.lead_status, lead.location_status) == (97, "HOT", "listed")
    assert lead.reason == REASON and lead.recommended_action.startswith("Contact the lead within 1 hour")
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
    saved = {"lead": {"name": "Rahul Sharma"}, "lead_id": 3}
    assert "offer a callback" in status_block(saved)
    assert "Still useful" not in status_block(saved)
    assert "offer a callback" not in status_block({**saved, "callback_status": "requested"})


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
    # TestClient runs the background task before returning, as the server does after the response
    assert (lead.reason, lead.reason_source) == (REASON, "llm")


def test_refresh_reason_skips_a_lead_whose_score_changed(ops):
    raw = {"name": "Rahul Sharma", "phone": "9876543210", "city": "Lucknow", "system_size_kw": 5}
    lead = service.create_lead(ops.engine, raw, areas=AREAS, reason_now=False).lead
    assert lead.reason_source == "template"

    class ChangesScoreFirst(FakeLLM):
        def invoke(self, messages):
            service.update_lead(ops.engine, lead.id, {"email": "rahul.sharma@example.com"}, reason_now=False)
            return super().invoke(messages)

    assert service.refresh_reason(ops.engine, lead.id, ChangesScoreFirst(REASON)) is False
    assert service.get_lead(ops.engine, lead.id).reason_source == "template"

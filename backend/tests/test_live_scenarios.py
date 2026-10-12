"""scripts/live_scenarios.py: the checks and the runner, on canned responses (no server, no network)."""
from app.agent.welcome import NUMBERS_CONFIRM_TEXT, REDIRECT_TEXT
from app.rag.answer import LLM_DOWN_REPLY
from scripts import live_scenarios as live

KB = ["Offering: Vantis HomeStart 3 / 3 kW rooftop solution", "Timing varies with project and site conditions."]


def response(reply, lead=None, steps=()):
    return {"session_id": "s-1", "reply": reply, "lead": lead, "steps": list(steps), "sources": [], "degraded": False}


def ok(check, resp, ctx=None):
    _, fn = check
    return fn(resp, ctx or live.Context(KB, ["How long does installation take?"]))


def test_number_checks():
    assert ok(live.no_invented_numbers(), response("HomeStart 3 is a reference option."))
    assert not ok(live.no_invented_numbers(), response("Installation takes 2 weeks."))
    assert ok(live.no_invented_numbers(), response("Got it, ₹4,000."), live.Context(KB, ["my bill is 4k"]))
    assert ok(live.no_amounts(), response(f"Timing varies. {NUMBERS_CONFIRM_TEXT}"))
    assert not ok(live.no_amounts(), response("About ₹2 lakh."))
    assert ok(live.team_confirms(), response(NUMBERS_CONFIRM_TEXT))


def test_question_lead_and_wording_checks():
    assert ok(live.exactly_one_question(), response("Which city is your home in?"))
    assert not ok(live.at_most_one_question(), response("Which city? And your bill?"))
    assert ok(live.no_lead(), response("Hi", lead={"id": None, "known_fields": []}))
    assert ok(live.lead_saved(), response("Saved", lead={"id": 3, "known_fields": ["phone"]}))
    assert ok(live.known("phone"), response("", lead={"id": 3, "known_fields": ["phone"]}))
    assert ok(live.not_known("email"), response("", lead=None))
    assert ok(live.city_is("Delhi"), response("", lead={"city": "Delhi"}))
    assert not ok(live.no_saved_claim(), response("Your details are saved."))
    assert not ok(live.plain_service_words(), response("Kota is a reference location."))
    assert ok(live.plain_service_words(), response("Kota is covered [KB Section 10: Reference Service Locations]."))
    assert ok(live.cites(10), response("Covered [KB Section 10: Reference Service Locations]."))


def test_tool_redirect_and_provider_checks():
    city = {"node": "tool", "name": "check_service_area", "detail": "Gurgaon: listed", "data": {}}
    assert ok(live.city_checked_listed(), response("", steps=[city]))
    support = {"node": "tool", "name": "create_support_request", "detail": "saved", "data": {}}
    assert ok(live.support_saved(), response("Your request id is 4.", steps=[support]))
    assert not ok(live.support_saved(), response("Your request is saved.", steps=[support]))
    assert ok(live.is_redirect(), response(REDIRECT_TEXT)) and not ok(live.not_redirect(), response(REDIRECT_TEXT))
    assert not ok(live.not_down(), response(LLM_DOWN_REPLY))
    groq = {"node": "agent", "detail": "groq", "data": {"provider": "groq"}}
    gemini = {"node": "agent", "detail": "gemini", "data": {"provider": "gemini"}}
    assert ok(live.answered_by_groq(), response("", steps=[groq]))
    assert not ok(live.answered_by_groq(), response("", steps=[groq, gemini]))


def test_maintenance_free_check():
    _, fn = live.NOT_FREE
    assert not fn(response("Yes, maintenance is free."), None)
    assert fn(response("The scope depends on your equipment terms."), None)


def test_runner_keeps_the_session_pauses_and_checks_every_turn():
    sent, paused = [], []
    replies = iter([response("Which city is your home in?"), response("Thanks. Your details are saved.")])

    def post(body):
        sent.append(body)
        return next(replies)

    scenario = live.Scenario("X", "edge", "two turns", [
        live.Turn("I want solar", [live.exactly_one_question()]),
        live.Turn("Delhi", [live.no_saved_claim()]),
    ])
    results = live.run_scenario(scenario, post, KB, 7, sleep=paused.append)
    assert sent == [{"message": "I want solar"}, {"message": "Delhi", "session_id": "s-1"}]
    assert paused == [7]
    assert results[0].failed == [] and results[1].failed == ["no 'saved' claim"]
    assert "| X | 2 | Delhi |" in live.markdown(results) and "FAIL" in live.markdown(results)


def test_every_holdout_question_has_a_scenario_and_ids_are_unique():
    scenarios = live.scenarios_for("holdout")
    assert [s.id for s in scenarios] == [f"h{i:02d}" for i in range(1, 9)]
    ids = [s.id for s in live.EDGE + live.SLOW]
    assert len(ids) == len(set(ids))
    assert all(len(s.turns[0].checks) >= 3 for s in scenarios)


def test_llm_calls_seen_counts_understand_and_agent_steps():
    steps = [{"node": "understand", "data": {"provider": "gemini"}}, {"node": "retrieve", "data": {}},
             {"node": "agent", "data": {}}, {"node": "tool", "data": {}}, {"node": "agent", "data": {}}]
    assert live.llm_calls_seen({"steps": steps}) == 3


def test_the_number_check_allows_step_details_and_the_phone_message():
    # Live edge run: the app's guard allowed these (tool result, validation message); the script could not see them
    support = {"node": "tool", "name": "create_support_request", "detail": "saved: request 1", "data": {}}
    assert ok(live.no_invented_numbers(), response("I have saved your support request with ID 1.", steps=[support]))
    phone = "That doesn't look like a 10-digit Indian mobile number (it should start with 6, 7, 8 or 9)."
    assert ok(live.no_invented_numbers(), response(f"{phone} Could you share it again?"))

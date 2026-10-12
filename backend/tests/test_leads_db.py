"""Lead service against the Neon test branch (TEST_DATABASE_URL). Fake LLM; no real LLM calls."""
import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from app.leads import validators as v
from app.leads.models import Callback, Lead, SupportRequest
from app.leads.scoring import HOT_ACTION
from app.leads.service import (
    CALLBACK_MESSAGE,
    CALLBACK_UPDATED_MESSAGE,
    LeadNotFound,
    create_lead,
    create_support_request,
    get_lead,
    list_leads,
    schedule_callback,
    update_lead,
)
from tests.leads_data import AREAS, EXPLORER, RAHUL
from tests.test_answer import FakeLLM, ReasonLLM

REASON = "Ready buyer with a clear 5 kW requirement and full contact details."


@pytest.fixture
def db(leads_db):
    """The test database with empty lead tables (emptied again after the test)."""
    return leads_db


@pytest.fixture
def llm():
    return ReasonLLM(REASON)


def count(engine, model) -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(model))


def test_init_db_creates_the_lead_tables(db):
    tables = set(inspect(db).get_table_names())
    assert {"leads", "callbacks", "support_requests"} <= tables


def test_round_trip_create_update_rescore_read(db, llm):
    # Rahul in Pune: unverified city, capped at WARM
    created = create_lead(db, {**RAHUL, "city": "Pune"}, session_id="s-1", llm=llm, areas=AREAS)
    lead = created.lead
    assert created.missing_fields == [] and created.errors == {}
    assert (lead.lead_score, lead.lead_status, lead.location_status) == (74, "WARM", "unverified")
    assert lead.score_breakdown["raw_score"] == 82
    # A WARM reason ends with what keeps it from HOT; for a capped lead, the cap's cause (T12a). T12b: code
    # writes that ending once, after the LLM's part (the fake's own ", not HOT yet: ..." is cut off)
    expected = "Ready buyer with a clear 5 kW requirement and full contact details; not HOT yet: " \
               "city not in the listed service areas."
    assert (lead.reason, lead.reason_source) == (expected, "llm")
    assert "Verify service availability for Pune." in lead.recommended_action
    assert (lead.phone, lead.monthly_bill_inr, lead.system_size_kw) == ("+919876543210", 4000, 5.0)
    assert lead.customer_type == "residential" and lead.session_id == "s-1"
    assert lead.created_at and lead.updated_at
    assert len(llm.calls) == 1

    # A new bill changes nothing in the breakdown ("monthly bill given"), so no new reason is written
    same = update_lead(db, lead.id, {"monthly_bill_inr": "5k"}, llm=llm, areas=AREAS).lead
    assert (same.monthly_bill_inr, same.lead_score, same.lead_status) == (5000, 74, "WARM")
    assert len(llm.calls) == 1

    # Moving to Lucknow: listed city, re-scored to HOT, new reason
    moved = update_lead(db, lead.id, {"city": "lucknow"}, llm=llm, areas=AREAS).lead
    assert (moved.city, moved.city_input, moved.location_status) == ("Lucknow", "lucknow", "listed")
    assert (moved.lead_score, moved.lead_status) == (97, "HOT")
    assert moved.recommended_action == HOT_ACTION
    assert len(llm.calls) == 2
    assert moved.updated_at >= lead.updated_at

    read = get_lead(db, lead.id)
    assert read.model_dump() == moved.model_dump()
    assert get_lead(db, 999) is None


def test_list_filters_and_sorts(db, llm):
    rahul = create_lead(db, RAHUL, llm=llm, areas=AREAS).lead  # HOT 97
    explorer = create_lead(db, EXPLORER, llm=llm, areas=AREAS).lead  # COLD 36
    newest, total = list_leads(db)
    assert total == 2 and [x.id for x in newest] == [explorer.id, rahul.id]
    by_score, _ = list_leads(db, sort="score")
    assert [x.lead_score for x in by_score] == [97, 36]
    hot, hot_total = list_leads(db, status="HOT")
    assert hot_total == 1 and hot[0].id == rahul.id
    assert list_leads(db, status="WARM") == ([], 0)
    with pytest.raises(ValueError):
        list_leads(db, status="LUKEWARM")
    with pytest.raises(ValueError):
        list_leads(db, sort="name")


def test_one_lead_per_session(db, llm):
    first = create_lead(db, {**RAHUL, "email": None}, session_id="s-2", llm=llm, areas=AREAS).lead
    second = create_lead(db, {"email": "rahul.sharma@example.com"}, session_id="s-2", llm=llm, areas=AREAS).lead
    assert second.id == first.id
    assert second.email == "rahul.sharma@example.com"
    assert count(db, Lead) == 1


def test_missing_details_save_nothing(db, llm):
    result = create_lead(db, {"name": "Rahul Sharma", "city": "Lucknow", "phone": "12345"}, llm=llm, areas=AREAS)
    assert result.lead is None
    assert result.missing_fields == ["phone or email", "requirement, monthly bill, usage, system size or property type"]
    assert "phone" in result.errors  # the invalid number is reported, never stored
    assert count(db, Lead) == 0
    assert llm.calls == []


def test_invalid_value_is_reported_but_the_lead_is_saved(db, llm):
    result = create_lead(db, {**RAHUL, "phone": "12345"}, llm=llm, areas=AREAS)
    assert result.lead.phone is None and result.lead.email == "rahul.sharma@example.com"
    assert set(result.errors) == {"phone"}


def test_llm_failure_still_saves_with_the_template_reason(db):
    result = create_lead(db, RAHUL, llm=FakeLLM(error=RuntimeError("down")), areas=AREAS)
    assert result.lead.reason_source == "template"
    assert result.lead.reason.startswith("HOT: ready to buy")


def test_update_of_a_missing_lead(db, llm):
    with pytest.raises(LeadNotFound):
        update_lead(db, 999, {"city": "Delhi"}, llm=llm, areas=AREAS)


def test_callback_needs_a_lead(db):
    with pytest.raises(LeadNotFound):
        schedule_callback(db, 999, "tomorrow evening")
    assert count(db, Callback) == 0


def test_callback_is_requested_never_booked(db, llm):
    lead = create_lead(db, RAHUL, llm=llm, areas=AREAS).lead
    result = schedule_callback(db, lead.id, "tomorrow evening", notes="after 6 pm")
    assert result.callback.status == "requested"
    assert result.callback.preferred_time == "tomorrow evening"
    assert result.message == CALLBACK_MESSAGE
    assert "booked" not in result.message.lower() and "confirmed" not in result.message.lower()
    read = get_lead(db, lead.id)
    assert [c.preferred_time for c in read.callbacks] == ["tomorrow evening"]
    # The time the customer asked for replaces "within 1 hour" (T12a)
    assert read.recommended_action == "Call tomorrow evening, as requested, and offer a site assessment."
    # The callback stays in the action after the lead is re-scored
    updated = update_lead(db, lead.id, {"ownership": "own"}, llm=llm, areas=AREAS).lead
    assert updated.recommended_action == "Call tomorrow evening, as requested, and offer a site assessment."
    assert HOT_ACTION not in updated.recommended_action
    with pytest.raises(v.InvalidValue):
        schedule_callback(db, lead.id, "   ")


def test_a_lead_keeps_one_callback_request_and_a_new_time_updates_it(db, llm):
    lead = create_lead(db, RAHUL, llm=llm, areas=AREAS).lead
    first = schedule_callback(db, lead.id, "4pm")
    second = schedule_callback(db, lead.id, "day after tomorrow, 4pm")
    assert (first.updated, first.message) == (False, CALLBACK_MESSAGE)
    assert (second.updated, second.message) == (True, CALLBACK_UPDATED_MESSAGE)
    assert second.callback.id == first.callback.id and second.callback.status == "requested"
    read = get_lead(db, lead.id)
    assert [(c.preferred_time, c.status) for c in read.callbacks] == [("day after tomorrow, 4pm", "requested")]
    assert read.recommended_action == "Call day after tomorrow, 4pm, as requested, and offer a site assessment."
    assert count(db, Callback) == 1


def test_support_requests_are_not_leads(db):
    result = create_support_request(
        db,
        {"name": "Anil Kumar", "phone": "98765 11111", "issue": "Inverter shows an error light", "project_details": "3 kW, 2024"},
        session_id="s-3",
    )
    assert result.id is not None and result.missing_fields == []
    assert count(db, SupportRequest) == 1
    assert count(db, Lead) == 0
    with Session(db) as session:
        stored = session.get(SupportRequest, result.id)
        assert (stored.phone, stored.status, stored.session_id) == ("+919876511111", "open", "s-3")


def test_support_request_needs_name_contact_and_issue(db):
    result = create_support_request(db, {"name": "Anil Kumar", "phone": "123"})
    assert result.id is None
    assert result.missing_fields == ["phone or email", "issue"]
    assert set(result.errors) == {"phone"}
    assert count(db, SupportRequest) == 0


def test_logs_hold_no_personal_details(db, llm, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="app.leads"):
        lead = create_lead(db, RAHUL, llm=llm, areas=AREAS).lead
        schedule_callback(db, lead.id, "tomorrow evening")
        create_support_request(db, {"name": "Anil Kumar", "email": "anil@example.com", "issue": "No power"})
    logged = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    for secret in ("Rahul", "9876543210", "rahul.sharma", "Anil", "anil@example.com", "tomorrow evening"):
        assert secret not in logged
    created = next(r for r in caplog.records if r.getMessage() == "lead created")
    assert (created.lead_id, created.lead_status, created.lead_score) == (lead.id, "HOT", 97)

"""Sales API on the Neon test branch (TEST_DATABASE_URL), fake reason model, fixed service-area list."""
import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.api import sales
from app.core.config import get_settings
from app.core.db import init_db
from app.leads import service
from app.leads.models import Callback, Lead
from app.leads.schemas import LeadData
from app.main import DB_DOWN, app
from tests.leads_data import AREAS, EXPLORER, RAHUL
from tests.test_answer import FakeLLM

KEY = "test-sales-key-0123456789"
REASON = "Ready buyer with a clear 5 kW requirement and full contact details."
LIST_FIELDS = {"id", "name", "lead_score", "lead_status", "city", "location_status", "requirement",
               "system_size_kw", "monthly_bill_inr", "recommended_action", "created_at"}


@pytest.fixture
def llm():
    return FakeLLM(REASON)


@pytest.fixture
def db(test_engine):
    init_db(test_engine)
    with test_engine.begin() as conn:
        conn.execute(text("TRUNCATE leads, callbacks, support_requests RESTART IDENTITY CASCADE"))
    return test_engine


@pytest.fixture
def api(client, monkeypatch, db, llm, use_engine):
    """The test client with the key, the test branch, the fake reason model and the fixed city list."""
    monkeypatch.setattr(get_settings(), "sales_api_key", KEY)
    use_engine(db)
    app.dependency_overrides[sales.get_reason_llm] = lambda: llm
    app.dependency_overrides[sales.get_lead_areas] = lambda: AREAS
    client.headers["X-API-Key"] = KEY
    return client


def add_lead(db, llm, raw, **kwargs):
    return service.create_lead(db, raw, llm=llm, areas=AREAS, **kwargs).lead


def count(db, model) -> int:
    with Session(db) as session:
        return session.scalar(select(func.count()).select_from(model))


# ---------- GET /api/leads ----------

def test_list_shows_the_dashboard_fields_newest_first(api, db, llm):
    rahul = add_lead(db, llm, RAHUL)
    explorer = add_lead(db, llm, EXPLORER)
    body = api.get("/api/leads").json()
    assert (body["total"], body["limit"], body["offset"]) == (2, 50, 0)
    assert [item["id"] for item in body["items"]] == [explorer.id, rahul.id]
    assert set(body["items"][1]) == LIST_FIELDS  # no phone, email or session_id in the list
    item = body["items"][1]
    assert (item["name"], item["lead_score"], item["lead_status"], item["city"]) == ("Rahul Sharma", 97, "HOT", "Lucknow")
    assert item["recommended_action"] == rahul.recommended_action


def test_list_filters_sorts_and_pages(api, db, llm):
    rahul = add_lead(db, llm, RAHUL)  # HOT 97
    explorer = add_lead(db, llm, EXPLORER)  # COLD 36
    hot = api.get("/api/leads", params={"status": "HOT"}).json()
    assert (hot["total"], [i["id"] for i in hot["items"]]) == (1, [rahul.id])
    assert api.get("/api/leads", params={"status": "WARM"}).json() == {"items": [], "total": 0, "limit": 50, "offset": 0}
    by_score = api.get("/api/leads", params={"sort": "score"}).json()
    assert [i["lead_score"] for i in by_score["items"]] == [97, 36]
    first = api.get("/api/leads", params={"limit": 1}).json()
    second = api.get("/api/leads", params={"limit": 1, "offset": 1}).json()
    assert [i["id"] for i in first["items"] + second["items"]] == [explorer.id, rahul.id]
    assert first["total"] == second["total"] == 2


@pytest.mark.parametrize(
    "params, field",
    [({"status": "LUKEWARM"}, "status"), ({"sort": "name"}, "sort"), ({"limit": 0}, "limit"),
     ({"limit": 201}, "limit"), ({"offset": -1}, "offset")],
)
def test_bad_list_parameters_are_422_with_a_message(api, params, field):
    response = api.get("/api/leads", params=params)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Some values are not valid." and set(body["errors"]) == {field}


# ---------- GET /api/leads/{id} ----------

def test_detail_has_every_field_breakdown_and_callbacks_but_no_session_id(api, db, llm):
    lead = add_lead(db, llm, RAHUL, session_id=str(uuid.uuid4()))
    service.schedule_callback(db, lead.id, "tomorrow evening")
    body = api.get(f"/api/leads/{lead.id}").json()
    assert "session_id" not in body
    assert set(LeadData.model_fields) <= set(body)
    assert (body["lead_score"], body["lead_status"], body["phone"]) == (97, "HOT", "+919876543210")
    assert [r["rule"] for r in body["score_breakdown"]["rules"]] == [
        "intent", "requirement", "location", "timeline", "contact", "property"]
    assert body["reason"] == REASON and body["reason_source"] == "llm"
    assert body["recommended_action"].endswith("Customer asked for a callback: tomorrow evening.")
    assert [(c["preferred_time"], c["status"]) for c in body["callbacks"]] == [("tomorrow evening", "requested")]
    assert body["created_at"] and body["updated_at"]


def test_unknown_lead_is_404_and_a_bad_id_is_422(api):
    response = api.get("/api/leads/999999")
    assert response.status_code == 404
    assert response.json()["detail"] == "No lead with id 999999."
    bad = api.get("/api/leads/abc")
    assert bad.status_code == 422 and set(bad.json()["errors"]) == {"lead_id"}


# ---------- POST /api/leads ----------

def test_create_scores_a_complete_lead_and_writes_the_reason_after(api, db, llm):
    response = api.post("/api/leads", json=RAHUL)  # raw values: "₹4,000", "5 kW", "9876543210"
    assert response.status_code == 201
    created = response.json()
    assert (created["lead_score"], created["lead_status"], created["location_status"]) == (97, "HOT", "listed")
    assert (created["phone"], created["monthly_bill_inr"], created["system_size_kw"]) == ("+919876543210", 4000, 5.0)
    assert created["reason_source"] == "template" and "session_id" not in created
    later = api.get(f"/api/leads/{created['id']}").json()  # the background task has run
    assert (later["reason"], later["reason_source"]) == (REASON, "llm")
    assert len(llm.calls) == 1


def test_create_refuses_an_incomplete_lead_and_saves_nothing(api, db, llm):
    response = api.post("/api/leads", json={"name": "Rahul Sharma", "city": "Lucknow"})
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "The lead is missing required details."
    assert body["missing_fields"] == ["phone or email", "requirement, monthly bill, usage, system size or property type"]
    assert count(db, Lead) == 0 and llm.calls == []


def test_create_refuses_invalid_values_with_field_messages(api, db):
    response = api.post("/api/leads", json={**RAHUL, "phone": "12345", "monthly_bill_inr": "-50"})
    assert response.status_code == 422
    errors = response.json()["errors"]
    assert set(errors) == {"phone", "monthly_bill_inr"}
    assert "Value error" not in errors["phone"] and "12345" not in str(response.json())  # input not echoed
    assert count(db, Lead) == 0


def test_create_refuses_unknown_fields(api, db):
    response = api.post("/api/leads", json={**RAHUL, "favourite_colour": "blue"})
    assert response.status_code == 422
    assert response.json()["errors"] == {"favourite_colour": "Unknown field."}
    assert api.post("/api/leads", content="not json", headers={"Content-Type": "application/json"}).status_code == 422
    assert count(db, Lead) == 0


# ---------- POST /api/callback ----------

def test_callback_is_logged_as_requested(api, db, llm):
    lead = add_lead(db, llm, RAHUL)
    response = api.post("/api/callback", json={"lead_id": lead.id, "preferred_time": " tomorrow  evening ",
                                               "notes": "after 6 pm"})
    assert response.status_code == 201
    body = response.json()
    assert (body["lead_id"], body["status"], body["preferred_time"], body["notes"]) == (
        lead.id, "requested", "tomorrow evening", "after 6 pm")
    assert body["message"] == service.CALLBACK_MESSAGE
    detail = api.get(f"/api/leads/{lead.id}").json()
    assert detail["recommended_action"].endswith("Customer asked for a callback: tomorrow evening.")


def test_callback_for_an_unknown_lead_is_404_and_saves_nothing(api, db):
    response = api.post("/api/callback", json={"lead_id": 999999, "preferred_time": "tomorrow"})
    assert response.status_code == 404
    assert response.json()["detail"] == "No lead with id 999999."
    assert count(db, Callback) == 0


@pytest.mark.parametrize(
    "payload, field",
    [({"lead_id": 1, "preferred_time": "   "}, "preferred_time"), ({"lead_id": 0, "preferred_time": "today"}, "lead_id"),
     ({"lead_id": 1, "preferred_time": "x" * 201}, "preferred_time"), ({"lead_id": 1}, "preferred_time")],
)
def test_bad_callback_requests_are_422(api, db, payload, field):
    response = api.post("/api/callback", json=payload)
    assert response.status_code == 422 and set(response.json()["errors"]) == {field}
    assert count(db, Callback) == 0


def test_blank_time_message_is_plain(api):
    errors = api.post("/api/callback", json={"lead_id": 1, "preferred_time": " "}).json()["errors"]
    assert errors == {"preferred_time": "The preferred time is empty."}


# ---------- GET /api/support-requests ----------

def test_support_requests_are_listed_newest_first_without_session_id(api, db):
    first = service.create_support_request(
        db, {"name": "Anil Kumar", "phone": "98765 11111", "issue": "Inverter error light"}, session_id="s-1").id
    second = service.create_support_request(db, {"name": "Meena Rao", "email": "meena@example.com", "issue": "No output"}).id
    body = api.get("/api/support-requests").json()
    assert body["total"] == 2 and [i["id"] for i in body["items"]] == [second, first]
    assert "session_id" not in body["items"][1]
    assert (body["items"][1]["phone"], body["items"][1]["status"]) == ("+919876511111", "open")
    assert api.get("/api/support-requests", params={"limit": 1, "offset": 1}).json()["items"][0]["id"] == first


# ---------- database down (no test branch needed) ----------

def test_database_down_is_503_json(client, monkeypatch, use_engine, unreachable_engine):
    monkeypatch.setattr(get_settings(), "sales_api_key", KEY)
    use_engine(unreachable_engine)
    headers = {"X-API-Key": KEY}
    for response in (
        client.get("/api/leads", headers=headers),
        client.get("/api/leads/1", headers=headers),
        client.get("/api/support-requests", headers=headers),
        client.post("/api/callback", headers=headers, json={"lead_id": 1, "preferred_time": "today"}),
    ):
        assert response.status_code == 503
        assert response.json()["detail"] == DB_DOWN and response.json()["request_id"]

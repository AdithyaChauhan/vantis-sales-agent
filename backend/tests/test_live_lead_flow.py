"""Live lead flow: real models, the real KB on main (read-only), every lead write on the Neon test branch.

Run: python -m pytest -m live -s tests/test_live_lead_flow.py
Prints each reply with its steps, node times and the LLM calls used.
"""
import re
import time
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text

from app.agent.graph import build_agent, get_agent
from app.core.config import get_settings
from app.core.db import get_engine, init_db
from app.leads import service
from app.main import app
from app.rag.retriever import count_kb_chunks

pytestmark = pytest.mark.live

SCORE_WORDS_RE = re.compile(r"\b(HOT|WARM|COLD|score|lead status)\b", re.IGNORECASE)
BOOKED_RE = re.compile(r"\b(booked|is confirmed|has been confirmed|is scheduled|has been scheduled)\b", re.IGNORECASE)
NAME_AND_PHONE_RE = re.compile(r"\bname\b.*\b(phone|number|mobile)\b|\b(phone|number|mobile)\b.*\bname\b",
                               re.IGNORECASE | re.DOTALL)
LLM_CALLS = {"total": 0}


@pytest.fixture
def lead_agent(test_engine):
    """The real agent writing leads to the test branch; skips without the database, keys or KB."""
    settings = get_settings()
    if not (settings.database_url and settings.google_api_key):
        pytest.skip("DATABASE_URL and GOOGLE_API_KEY are needed")
    if not count_kb_chunks(get_engine()):
        pytest.skip("kb_chunks is empty; run python -m app.rag.ingest first")
    init_db(test_engine)
    with test_engine.begin() as conn:
        conn.execute(text("TRUNCATE leads, callbacks, support_requests RESTART IDENTITY CASCADE"))
    agent = build_agent(get_engine(), InMemorySaver(), leads_engine=test_engine)
    app.dependency_overrides[get_agent] = lambda: agent
    yield agent
    time.sleep(4)  # stay under 15 requests a minute per Gemini model


def ask(client, message: str, session_id: str) -> dict:
    start = time.perf_counter()
    response = client.post("/api/chat", json={"message": message, "session_id": session_id})
    total_ms = round((time.perf_counter() - start) * 1000)
    assert response.status_code == 200
    body = response.json()
    llm = sum(1 for s in body["steps"] if s["node"] in ("understand", "agent"))
    llm += sum(1 for s in body["steps"] if s.get("name") == "create_lead" and s["detail"].startswith("saved"))
    LLM_CALLS["total"] += llm
    times = ", ".join(
        f"{s['node']}{':' + s['name'] if s.get('name') else ''} {s['data'].get('latency_ms')} ms"
        for s in body["steps"] if s["data"].get("latency_ms") is not None
    )
    print(f"\n>>> {message}\n{body['reply']}\nsources: {[s['section'] for s in body['sources']]} | lead: {body['lead']}"
          f"\nsteps: {' → '.join(s['node'] + (':' + s['name'] if s.get('name') else '') + ' (' + s['detail'] + ')' for s in body['steps'])}"
          f"\ntimes: {times} | total {total_ms} ms | LLM calls {llm} (running total {LLM_CALLS['total']})")
    assert not SCORE_WORDS_RE.search(body["reply"]), "the reply mentions a score or status"
    assert not BOOKED_RE.search(body["reply"]), "the reply claims a booking"
    return body


def test_demo_script_creates_a_hot_lead_with_a_requested_callback(client, lead_agent, test_engine):
    session = str(uuid.uuid4())
    ask(client, "What solar systems do you offer for homes?", session)
    second = ask(client, "I'm interested in a 5 kW system for my house in Lucknow. My bill is about ₹4,000 a month "
                         "and I'd like it installed next month.", session)
    assert NAME_AND_PHONE_RE.search(second["reply"]), "turn 2 should ask for name and phone"
    third = ask(client, "Rahul Sharma, 9876543210, rahul.sharma@example.com", session)
    assert any(s.get("name") == "create_lead" and s["detail"].startswith("saved") for s in third["steps"])
    fourth = ask(client, "Yes, call me tomorrow evening.", session)
    assert any(s.get("name") == "schedule_callback" for s in fourth["steps"])

    leads, total = service.list_leads(test_engine)
    assert total == 1
    lead = service.get_lead(test_engine, leads[0].id)
    print(f"\nLEAD {lead.id}: {lead.lead_score} {lead.lead_status} ({lead.location_status}) | reason "
          f"[{lead.reason_source}]: {lead.reason} | action: {lead.recommended_action} | callbacks: "
          f"{[(c.preferred_time, c.status) for c in lead.callbacks]}")
    assert lead.lead_status == "HOT" and lead.lead_score >= 75
    assert lead.reason and lead.recommended_action
    assert [c.status for c in lead.callbacks] == ["requested"]
    assert "tomorrow evening" in lead.callbacks[0].preferred_time.lower()


def test_yes_please_connect_me_asks_for_name_and_phone(client, lead_agent):
    session = str(uuid.uuid4())
    ask(client, "Do you provide installation in Delhi?", session)
    body = ask(client, "Yes please, connect me", session)
    assert NAME_AND_PHONE_RE.search(body["reply"]), "should ask for name and phone"
    assert "HomeStart" not in body["reply"], "listed the offerings instead of asking for contact details"
    assert not any(s["node"] == "retrieve" for s in body["steps"])

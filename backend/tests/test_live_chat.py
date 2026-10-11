"""Live agent checks: real Gemini/Groq models and the real KB on Neon (read-only).

Memory is in-process (InMemorySaver), so nothing is written to the main database; the Postgres
checkpointer itself is covered by tests/test_agent_memory_db.py on the test branch.
Run: python -m pytest -m live -s   (prints each reply, its steps and the LLM calls used)
"""
import re
import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_agent, get_agent
from app.agent.welcome import WELCOME_TEXT
from app.core.config import get_settings
from app.core.db import get_engine
from app.main import app
from app.rag.answer import cited_sections
from app.rag.retriever import count_kb_chunks

pytestmark = pytest.mark.live

PACKAGES = ["HomeStart 3", "HomePro 5", "HomeMax 10", "Business 15", "Business 25"]
# A price figure: a currency word or sign, lakh/crore, or a number with thousands separators.
PRICE_RE = re.compile(r"₹|\bRs\.?|\bINR\b|\blakhs?\b|\bcrores?\b|\d{1,3}(?:,\d{2,3})+", re.IGNORECASE)
NUMBER_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|twenty-five|thirty"
YEARS_RE = re.compile(rf"\b(?:\d+|{NUMBER_WORDS})[\s-]*(?:years?|yrs?)\b", re.IGNORECASE)
GREETING_RE = re.compile(r"\s*(?:hello|hi|hey|greetings)\b", re.IGNORECASE)
NOT_SERVED_RE = re.compile(r"\b(?:do not|don't|does not|doesn't|cannot|can't)\s+(?:serve|install|operate|cover)", re.I)
CLAIMED_ACTION_RE = re.compile(
    r"\b(?:saved|booked|scheduled|passed (?:on|along)|forwarded|will call|will contact|"
    r"has been (?:notified|shared|sent)|have (?:notified|shared|sent|passed))\b",
    re.IGNORECASE,
)
LLM_CALLS = {"total": 0}


@pytest.fixture(autouse=True)
def live_agent(test_engine):
    """The real agent with in-process memory; skips without the database, keys or KB."""
    settings = get_settings()
    engine = get_engine()
    if not (settings.database_url and settings.google_api_key):
        pytest.skip("DATABASE_URL and GOOGLE_API_KEY are needed")
    if not count_kb_chunks(engine):
        pytest.skip("kb_chunks is empty; run python -m app.rag.ingest first")
    agent = build_agent(engine, InMemorySaver(), leads_engine=test_engine)  # lead writes: test branch only
    app.dependency_overrides[get_agent] = lambda: agent
    yield agent
    time.sleep(4)  # stay under 15 requests a minute per Gemini model


def ask(client, message: str, session_id: str | None = None) -> dict:
    payload = {"message": message} | ({"session_id": session_id} if session_id else {})
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    body = response.json()
    calls = sum(1 for s in body["steps"] if s["node"] in ("understand", "agent"))
    LLM_CALLS["total"] += calls
    steps = " → ".join(f"{s['node']}{':' + s['name'] if s.get('name') else ''} ({s['detail']})" for s in body["steps"])
    print(f"\n>>> {message}\n{body['reply']}\nsources: {[s['section'] for s in body['sources']]}\nsteps: {steps}"
          f"\nLLM calls: {calls} (running total {LLM_CALLS['total']})")
    return body


def tool_steps(body: dict, name: str) -> list[dict]:
    return [s for s in body["steps"] if s["node"] == "tool" and s.get("name") == name]


def test_q01_names_every_package(client):
    body = ask(client, "What solar systems do you provide?")
    assert not body["degraded"]
    missing = [p for p in PACKAGES if p not in body["reply"]]
    assert not missing, f"packages not named: {missing}"
    assert 8 in cited_sections(body["reply"])


def test_q02_price_question_gives_no_price(client):
    body = ask(client, "How much does a 5kW system cost?")
    assert not PRICE_RE.search(body["reply"]), "the reply contains a price"
    assert set(cited_sections(body["reply"])) & {5, 15, 8}


def test_q05_warranty_gives_no_duration(client):
    body = ask(client, "What warranty do you provide?")
    assert not YEARS_RE.search(body["reply"]), "the reply states a warranty duration"
    assert not GREETING_RE.match(body["reply"]), "the reply opens with a greeting"
    assert set(cited_sections(body["reply"])) & {11, 15}


def test_follow_up_uses_memory(client):
    first = ask(client, "Do you provide installation in Delhi?", session_id="live-memory")
    assert 10 in cited_sections(first["reply"])
    second = ask(client, "And in Kanpur?", session_id="live-memory")
    assert "Kanpur" in second["reply"]
    assert 10 in cited_sections(second["reply"]) or tool_steps(second, "check_service_area")
    assert "install" in second["reply"].lower() or "service" in second["reply"].lower()


def test_pune_is_checked_and_not_refused(client):
    body = ask(client, "Can you install solar panels in Pune?")
    checks = tool_steps(body, "check_service_area")
    assert checks and checks[0]["detail"].lower().startswith("pune: unverified")
    assert "listed" in body["reply"].lower()
    assert not NOT_SERVED_RE.search(body["reply"]), "the reply says Vantis doesn't serve Pune"


def test_hello_gets_the_welcome(client):
    body = ask(client, "hi")
    assert body["reply"] == WELCOME_TEXT
    assert [s["node"] for s in body["steps"]] == ["understand", "reply"]


def test_no_claimed_actions_after_yes_please(client):
    ask(client, "Do you provide installation in Delhi?", session_id="live-no-claims")
    body = ask(client, "Yes please, connect me", session_id="live-no-claims")
    assert any(s["node"] == "agent" for s in body["steps"]), "went to a fixed reply, not the agent"
    assert not CLAIMED_ACTION_RE.search(body["reply"]), "the reply claims an action that did not happen"


def test_groq_answers_when_gemini_fails(client, test_engine):
    settings = get_settings()
    if not (settings.groq_api_key and settings.fallback_model):
        pytest.skip("GROQ_API_KEY and FALLBACK_MODEL are needed")
    agent = build_agent(get_engine(), InMemorySaver(), main_model="gemini-model-does-not-exist",
                        leads_engine=test_engine)
    app.dependency_overrides[get_agent] = lambda: agent
    body = ask(client, "What is your installation process?")
    agent_steps = [s for s in body["steps"] if s["node"] == "agent"]
    assert agent_steps and all(s["data"]["provider"] == "groq" for s in agent_steps)
    assert body["reply"] and "<think>" not in body["reply"]
    assert cited_sections(body["reply"]), "Groq reply has no citation"

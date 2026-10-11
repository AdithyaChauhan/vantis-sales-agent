"""Live voice check: Adithya's 4-turn session from the T10 local run, in one session, with real models.

The KB is read from main; lead writes (if any) go to the Neon test branch; memory is in-process.
Run: python -m pytest -m live -s tests/test_live_voice.py
Prints each reply with its sources, step times, total time and LLM calls. Tone is judged by reading the
output; the asserts cover the safety rules only (collected per turn and checked at the end, so every turn runs).
"""
import time
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_agent, get_agent
from app.agent.understand import load_kb_hints
from app.core.config import get_settings
from app.core.db import get_engine
from app.main import app
from app.rag.answer import snippet
from app.rag.chunker import SERVICE_AREA_SECTION
from app.rag.retriever import count_kb_chunks
from tests.test_live_chat import NOT_SERVED_RE, PRICE_RE
from tests.test_live_lead_flow import KB_LABEL_RE, PAUSE_BETWEEN_TURNS_S, SCORE_WORDS_RE, TOOL_CITATION_RE

pytestmark = pytest.mark.live

LLM_CALLS = {"total": 0}


@pytest.fixture
def voice_agent(test_engine):
    """The real agent with in-process memory; skips without the database, keys or KB."""
    settings = get_settings()
    if not (settings.database_url and settings.google_api_key):
        pytest.skip("DATABASE_URL and GOOGLE_API_KEY are needed")
    if not count_kb_chunks(get_engine()):
        pytest.skip("kb_chunks is empty; run python -m app.rag.ingest first")
    agent = build_agent(get_engine(), InMemorySaver(), leads_engine=test_engine)
    app.dependency_overrides[get_agent] = lambda: agent
    yield agent
    time.sleep(4)


def ask(client, message: str, session_id: str, turns: list) -> dict:
    """Send one message (after a pause, except the first), print it, and note any broken safety rule."""
    if turns:
        time.sleep(PAUSE_BETWEEN_TURNS_S)
    start = time.perf_counter()
    response = client.post("/api/chat", json={"message": message, "session_id": session_id})
    total_ms = round((time.perf_counter() - start) * 1000)
    assert response.status_code == 200
    body = response.json()
    reply = body["reply"]
    calls = sum(1 for s in body["steps"] if s["node"] in ("understand", "agent"))
    LLM_CALLS["total"] += calls
    times = ", ".join(
        f"{s['node']}{':' + s['name'] if s.get('name') else ''} {s['data'].get('latency_ms')} ms"
        + (f" [{s['data'].get('model') or s['data'].get('provider')}]" if s["node"] in ("understand", "agent") else "")
        for s in body["steps"] if s["data"].get("latency_ms") is not None
    )
    sources = [f"{s['section']}: {s['snippet'][:60]}" for s in body["sources"]]
    print(f"\n>>> {message}\n{reply}\nsources: {sources}\nsteps: "
          f"{' → '.join(s['node'] + (':' + s['name'] if s.get('name') else '') + ' (' + s['detail'] + ')' for s in body['steps'])}"
          f"\ntimes: {times} | total {total_ms} ms | LLM calls {calls} (running total {LLM_CALLS['total']})")

    problems = []
    if PRICE_RE.search(reply):
        problems.append("states a price")
    if NOT_SERVED_RE.search(reply):
        problems.append("says Vantis doesn't serve a place")
    if SCORE_WORDS_RE.search(reply):
        problems.append("mentions a score or status")
    if TOOL_CITATION_RE.search(reply):
        problems.append("cites a tool")
    labels = KB_LABEL_RE.findall(reply)
    if len(labels) != len(set(labels)):
        problems.append("repeats a KB citation")
    if reply.count("?") > 1:
        problems.append("asks more than one question")
    turns.append({"message": message, "body": body, "problems": problems, "total_ms": total_ms})
    return body


def checked_area(body: dict) -> bool:
    return any(s["node"] == "tool" and s.get("name") == "check_service_area" for s in body["steps"])


def test_adithya_session_reads_naturally_and_keeps_the_rules(client, voice_agent):
    session = str(uuid.uuid4())
    turns: list[dict] = []
    ask(client, "what is the process of installing? what is the procedure?", session, turns)
    ask(client, "Bengaluru", session, turns)
    ask(client, "Let's say Hyderabad!", session, turns)
    ask(client, "5kb something like that maybe? what do you provide?", session, turns)

    summary = load_kb_hints(get_engine()).summaries.get(SERVICE_AREA_SECTION)
    for turn in turns[1:3]:  # the two city turns
        body = turn["body"]
        if not checked_area(body):
            turn["problems"].append("no check_service_area call")
        area_sources = [s for s in body["sources"] if s["section"] == str(SERVICE_AREA_SECTION)]
        if summary and area_sources and area_sources[0]["snippet"] != snippet(summary["content"]):
            turn["problems"].append("the KB Section 10 source is not the summary chunk")
    print(f"\nTurn totals (ms): {[t['total_ms'] for t in turns]} | LLM calls: {LLM_CALLS['total']}")
    broken = {t["message"]: t["problems"] for t in turns if t["problems"]}
    assert not broken, broken

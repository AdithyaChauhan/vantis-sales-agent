"""POST /api/chat through the agent graph, with fake models and a fake search (no network)."""
import uuid

import psycopg
import pytest
from sqlalchemy.exc import OperationalError

from app.agent.graph import get_agent
from app.agent.welcome import WELCOME_TEXT
from app.main import app
from app.rag.answer import LLM_DOWN_REPLY
from tests.agent_fakes import FakeSearch, Scripted, answer, make_graph, understood

CITED = "We offer Plan A [KB Section 8: Catalog]."


@pytest.fixture
def use_agent():
    """Return a function that makes the app use a graph built from the given fakes."""

    def _use(understand_llm=None, agent_llm=None, search=None, graph=None):
        if graph is None:
            graph, _ = make_graph(
                understand_llm or Scripted(understood(["question"], "catalog")),
                agent_llm or Scripted(answer(CITED)),
                search=search,
            )
        app.dependency_overrides[get_agent] = lambda: graph
        return graph

    return _use


def post(client, message, **extra):
    return client.post("/api/chat", json={"message": message, **extra})


def test_response_shape(client, use_agent):
    use_agent()
    response = post(client, "What do you offer?")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"session_id", "reply", "sources", "steps", "lead", "degraded"}
    assert uuid.UUID(body["session_id"])  # a new id when none was given
    assert body["reply"] == CITED
    assert body["lead"] is None and body["degraded"] is False
    assert [s["section"] for s in body["sources"]] == ["8"]
    assert set(body["sources"][0]) == {"section", "title", "page", "snippet"}
    assert [s["node"] for s in body["steps"]] == ["understand", "retrieve", "agent"]
    assert set(body["steps"][0]) == {"node", "name", "detail", "data"}


def test_session_id_is_echoed_and_remembered(client, use_agent):
    understand_llm = Scripted(understood(["question"], "q"))
    use_agent(understand_llm=understand_llm, agent_llm=Scripted(answer("A1"), answer("A2")))
    session_id = str(uuid.uuid4())
    assert post(client, "First question", session_id=session_id).json()["session_id"] == session_id
    post(client, "Second question", session_id=session_id)
    assert "Customer: First question" in understand_llm.calls[1][0].content


def test_tool_steps_have_a_name(client, use_agent):
    from tests.agent_fakes import tool_call

    use_agent(agent_llm=Scripted(tool_call("check_service_area", {"city": "Pune"}), answer("Pune: the team can check.")))
    steps = post(client, "Pune?").json()["steps"]
    tool = next(s for s in steps if s["node"] == "tool")
    assert (tool["name"], tool["detail"]) == ("check_service_area", "Pune: unverified")


def test_greeting_returns_the_welcome(client, use_agent):
    use_agent(understand_llm=Scripted(understood(["greeting"])))
    assert post(client, "hi").json()["reply"] == WELCOME_TEXT


def test_llm_failure_is_200_and_degraded(client, use_agent):
    use_agent(agent_llm=Scripted(RuntimeError("Gemini and Groq both down")))
    response = post(client, "What do you offer?")
    assert response.status_code == 200
    body = response.json()
    assert (body["reply"], body["degraded"], body["sources"]) == (LLM_DOWN_REPLY, True, [])


@pytest.mark.parametrize(
    "error",
    [OperationalError("SELECT 1", {}, Exception("down")), psycopg.OperationalError("down")],
)
def test_database_errors_are_503(client, use_agent, error):
    use_agent(search=FakeSearch(error=error))
    response = post(client, "Do you install in Delhi?")
    assert response.status_code == 503
    assert set(response.json()) == {"detail", "request_id"}


def test_no_database_configured_is_503(client):
    app.dependency_overrides[get_agent] = lambda: None
    assert post(client, "Hi").status_code == 503


def test_unexpected_error_is_still_200(client, use_agent):
    class Broken:
        def invoke(self, *args, **kwargs):
            raise KeyError("bug")

    use_agent(graph=Broken())
    response = post(client, "Hi")
    assert response.status_code == 200
    assert response.json()["degraded"] is True


@pytest.mark.parametrize("message", ["", "    ", "\x00\x07", "x" * 1001])
def test_invalid_messages_are_rejected(client, use_agent, message):
    understand_llm = Scripted(understood(["question"], "q"))
    use_agent(understand_llm=understand_llm)
    assert post(client, message).status_code == 422
    assert understand_llm.calls == []


def test_missing_message_is_rejected(client):
    assert client.post("/api/chat", json={"session_id": str(uuid.uuid4())}).status_code == 422


def test_1000_characters_after_trimming_is_accepted(client, use_agent):
    understand_llm = Scripted(understood(["question"], "q"))
    use_agent(understand_llm=understand_llm)
    assert post(client, "  " + "x" * 1000 + "  ").status_code == 200
    assert "x" * 1000 in understand_llm.calls[0][0].content


def test_control_characters_are_stripped(client, use_agent):
    understand_llm = Scripted(understood(["question"], "q"))
    use_agent(understand_llm=understand_llm)
    post(client, " Do you\x00 install\x1b in\nDelhi?\t ")
    assert "Do you install in\nDelhi?" in understand_llm.calls[0][0].content


def test_log_line_has_no_message_text(client, use_agent, caplog):
    import logging

    use_agent()
    with caplog.at_level(logging.INFO, logger="app.api.chat"):
        post(client, "my number is 9876543210")
    record = next(r for r in caplog.records if r.getMessage() == "chat")
    assert "9876543210" not in str(record.__dict__)
    assert record.nodes == ["understand", "retrieve", "agent"]


# ---------- T8b: session lock and lead summary ----------
import threading  # noqa: E402
import time  # noqa: E402

from app.api import chat as chat_api  # noqa: E402
from app.core.config import get_settings  # noqa: E402

SUMMARY = {"id": 7, "status": "HOT", "score": 97, "missing_fields": ["ownership"], "callback_status": "requested"}


class StateAgent:
    """A fake agent returning a fixed final state; optionally slow, recording overlapping turns."""

    def __init__(self, summary=None, delay=0.0):
        self.summary, self.delay = summary, delay
        self.active = self.max_active = 0
        self.guard = threading.Lock()

    def invoke(self, payload, config):
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self.guard:
            self.active -= 1
        return {"reply": "ok", "steps": [], "sources": [], "lead_summary": self.summary, "degraded": False}


def test_lead_summary_hides_status_and_score_by_default(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "chat_show_lead_score", False)
    app.dependency_overrides[get_agent] = lambda: StateAgent(SUMMARY)
    lead = post(client, "hi").json()["lead"]
    assert lead == {"id": 7, "status": None, "score": None, "missing_fields": ["ownership"],
                    "callback_status": "requested"}


def test_lead_summary_shows_status_and_score_for_the_demo(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "chat_show_lead_score", True)
    app.dependency_overrides[get_agent] = lambda: StateAgent(SUMMARY)
    assert post(client, "hi").json()["lead"] == SUMMARY


def test_no_lead_summary_is_null(client):
    app.dependency_overrides[get_agent] = lambda: StateAgent(None)
    assert post(client, "hi").json()["lead"] is None


def test_turns_of_one_session_never_overlap(client):
    agent = StateAgent(delay=0.3)
    app.dependency_overrides[get_agent] = lambda: agent
    same = {"session_id": str(uuid.uuid4())}
    threads = [threading.Thread(target=post, args=(client, "hi"), kwargs=same) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert agent.max_active == 1


def test_busy_session_gets_a_friendly_reply(client, monkeypatch):
    monkeypatch.setattr(chat_api, "LOCK_TIMEOUT_S", 0.1)
    app.dependency_overrides[get_agent] = lambda: StateAgent()
    session_id = str(uuid.uuid4())
    lock = chat_api.session_lock(session_id)
    lock.acquire()
    try:
        body = post(client, "hi", session_id=session_id).json()
    finally:
        lock.release()
    assert body["reply"] == chat_api.BUSY_REPLY and body["degraded"] is True


# ---------- T9: random session ids, rate limit, client IP ----------
from app.api import security  # noqa: E402
from app.core.rate_limit import RateLimiter  # noqa: E402


@pytest.mark.parametrize(
    "session_id",
    ["demo-1", "abc-123", "", str(uuid.uuid1()), "00000000-0000-0000-0000-000000000000",
     "12345678-1234-1234-1234-123456789abc"],  # the last one is version 1
)
def test_guessable_session_ids_are_refused(client, use_agent, session_id):
    understand_llm = Scripted(understood(["question"], "q"))
    use_agent(understand_llm=understand_llm)
    response = post(client, "hi", session_id=session_id)
    assert response.status_code == 422
    assert response.json()["errors"]["session_id"] == chat_api.SESSION_ID_ERROR
    assert understand_llm.calls == []


def test_a_uuid4_is_accepted_in_its_standard_form(client):
    app.dependency_overrides[get_agent] = lambda: StateAgent()
    session_id = str(uuid.uuid4())
    assert post(client, "hi", session_id=session_id.upper()).json()["session_id"] == session_id


def use_limits(monkeypatch, per_minute, per_day=1000):
    monkeypatch.setattr(security, "CHAT_LIMITER", RateLimiter([(per_minute, 60), (per_day, 86_400)]))


def test_over_the_minute_limit_is_429_with_retry_after(client, monkeypatch):
    use_limits(monkeypatch, per_minute=2)
    agent = StateAgent()
    app.dependency_overrides[get_agent] = lambda: agent
    assert [post(client, "hi").status_code for _ in range(2)] == [200, 200]
    response = post(client, "hi")
    assert response.status_code == 429
    body = response.json()
    assert set(body) == {"detail", "retry_after", "request_id"}
    assert 1 <= body["retry_after"] <= 60 and response.headers["Retry-After"] == str(body["retry_after"])
    assert body["detail"] == security.TOO_FAST.format(seconds=body["retry_after"])


def test_over_the_day_limit_says_so(client, monkeypatch):
    use_limits(monkeypatch, per_minute=100, per_day=1)
    app.dependency_overrides[get_agent] = lambda: StateAgent()
    post(client, "hi")
    response = post(client, "hi")
    assert response.status_code == 429
    assert response.json()["detail"] == security.DAY_LIMIT.format(wait="24 hours")
    assert int(response.headers["Retry-After"]) > 86_000


def test_invalid_requests_count_toward_the_limit(client, monkeypatch, use_agent):
    use_limits(monkeypatch, per_minute=2)
    use_agent()
    assert [post(client, "").status_code for _ in range(2)] == [422, 422]
    assert post(client, "What do you offer?").status_code == 429


def test_spoofed_forwarding_headers_do_not_bypass_the_limit(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", "")
    use_limits(monkeypatch, per_minute=2)
    app.dependency_overrides[get_agent] = lambda: StateAgent()
    statuses = [
        client.post("/api/chat", json={"message": "hi"},
                    headers={"X-Forwarded-For": f"10.0.0.{i}", "CF-Connecting-IP": f"10.9.0.{i}"}).status_code
        for i in range(3)
    ]
    assert statuses == [200, 200, 429]


def test_with_the_header_configured_each_client_has_its_own_limit(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", "CF-Connecting-IP")
    use_limits(monkeypatch, per_minute=1)
    app.dependency_overrides[get_agent] = lambda: StateAgent()

    def send(ip, forwarded="1.1.1.1"):
        return client.post("/api/chat", json={"message": "hi"},
                           headers={"CF-Connecting-IP": ip, "X-Forwarded-For": forwarded}).status_code

    assert send("203.0.113.7") == 200
    assert send("203.0.113.7", forwarded="2.2.2.2") == 429  # a new X-Forwarded-For changes nothing
    assert send("198.51.100.4") == 200  # another real client


def test_request_log_says_where_the_ip_came_from_but_never_the_ip(client, monkeypatch, caplog):
    import logging

    use_limits(monkeypatch, per_minute=1)
    app.dependency_overrides[get_agent] = lambda: StateAgent()
    with caplog.at_level(logging.INFO):
        post(client, "hi")
        post(client, "hi")  # 429
        monkeypatch.setattr(get_settings(), "client_ip_header", "CF-Connecting-IP")
        client.post("/api/chat", json={"message": "hi"}, headers={"CF-Connecting-IP": "203.0.113.7"})
    requests = [r for r in caplog.records if r.getMessage() == "request"]
    assert [(r.status, r.ip_source) for r in requests] == [(200, "fallback"), (429, "fallback"), (200, "header")]
    logged = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert "testclient" not in logged and "203.0.113.7" not in logged
    assert any(r.getMessage() == "chat rate limit reached" and r.limit == "minute" for r in caplog.records)

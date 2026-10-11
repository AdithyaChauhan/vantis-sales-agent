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
    assert post(client, "First question", session_id="abc-123").json()["session_id"] == "abc-123"
    post(client, "Second question", session_id="abc-123")
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
    assert client.post("/api/chat", json={"session_id": "x"}).status_code == 422


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

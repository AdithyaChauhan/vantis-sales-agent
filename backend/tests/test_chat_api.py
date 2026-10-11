"""POST /api/chat with a fake LLM. Most tests replace search(); the last one uses the test database."""
import uuid

import pytest

from app.api import chat as chat_api
from app.core.llm import main_llm, query_embedder
from app.main import app
from app.rag.answer import LLM_DOWN_REPLY, NO_CONTEXT_REPLY
from app.rag.ingest import ingest_sections
from app.rag.retriever import SearchResult
from tests.test_answer import CHUNKS, FakeLLM
from tests.test_kb_db import SECTIONS, WARRANTY, FakeEmbedder, mix

CITED = "HomePro 5 is a 5 kW option [KB Section 8: Product & Service Catalog]."


@pytest.fixture
def fake_llm():
    """Return a function that makes the app use a given fake LLM."""

    def _use(llm):
        app.dependency_overrides[main_llm] = lambda: llm
        return llm

    return _use


@pytest.fixture
def fake_search(monkeypatch):
    """Replace search() with a canned result and record the questions it was asked."""
    asked = []

    def _use(result):
        def _search(query, k=4, **kwargs):
            asked.append((query, k))
            return result

        monkeypatch.setattr(chat_api, "search", _search)
        return asked

    return _use


def post(client, message, **extra):
    return client.post("/api/chat", json={"message": message, **extra})


def test_response_shape(client, fake_llm, fake_search):
    llm = fake_llm(FakeLLM(CITED))
    asked = fake_search(SearchResult(chunks=CHUNKS, mode="hybrid"))
    response = post(client, "What do you offer?")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"session_id", "reply", "sources", "steps", "lead", "degraded"}
    assert uuid.UUID(body["session_id"])  # a new id when none was given
    assert body["reply"] == CITED
    assert body["lead"] is None and body["degraded"] is False
    assert [s["section"] for s in body["sources"]] == ["8"]
    assert set(body["sources"][0]) == {"section", "title", "page", "snippet"}
    assert [s["node"] for s in body["steps"]] == ["retrieve", "answer"]
    assert body["steps"][1]["data"]["provider"] == "gemini"
    assert asked == [("What do you offer?", 4)]
    assert len(llm.calls) == 1


def test_session_id_is_echoed(client, fake_llm, fake_search):
    fake_llm(FakeLLM(CITED))
    fake_search(SearchResult(chunks=CHUNKS))
    response = post(client, "Hi", session_id="abc-123")
    assert response.json()["session_id"] == "abc-123"


def test_no_context_makes_no_llm_call(client, fake_llm, fake_search):
    llm = fake_llm(FakeLLM(CITED))
    fake_search(SearchResult(chunks=CHUNKS, no_context=True))
    body = post(client, "Tell me a joke").json()
    assert llm.calls == []
    assert body["reply"] == NO_CONTEXT_REPLY
    assert body["sources"] == []


def test_llm_failure_is_200_and_degraded(client, fake_llm, fake_search):
    fake_llm(FakeLLM(error=RuntimeError("Gemini and Groq both down")))
    fake_search(SearchResult(chunks=CHUNKS))
    response = post(client, "What do you offer?")
    assert response.status_code == 200
    body = response.json()
    assert (body["reply"], body["degraded"], body["sources"]) == (LLM_DOWN_REPLY, True, [])


def test_unexpected_error_is_still_200(client, fake_llm, fake_search, monkeypatch):
    fake_llm(FakeLLM(CITED))
    fake_search(SearchResult(chunks=CHUNKS))

    def broken(*args, **kwargs):
        raise KeyError("bug")

    monkeypatch.setattr(chat_api, "answer", broken)
    response = post(client, "Hi")
    assert response.status_code == 200
    assert response.json()["degraded"] is True


@pytest.mark.parametrize("message", ["", "    ", "\x00\x07", "x" * 1001])
def test_invalid_messages_are_rejected(client, fake_llm, fake_search, message):
    llm = fake_llm(FakeLLM(CITED))
    asked = fake_search(SearchResult(chunks=CHUNKS))
    assert post(client, message).status_code == 422
    assert asked == [] and llm.calls == []


def test_missing_message_is_rejected(client):
    assert client.post("/api/chat", json={"session_id": "x"}).status_code == 422


def test_1000_characters_after_trimming_is_accepted(client, fake_llm, fake_search):
    fake_llm(FakeLLM(CITED))
    asked = fake_search(SearchResult(chunks=CHUNKS))
    assert post(client, "  " + "x" * 1000 + "  ").status_code == 200
    assert asked[0][0] == "x" * 1000


def test_control_characters_are_stripped(client, fake_llm, fake_search):
    fake_llm(FakeLLM(CITED))
    asked = fake_search(SearchResult(chunks=CHUNKS))
    post(client, " Do you\x00 install\x1b in\nDelhi?\t ")
    assert asked[0][0] == "Do you install in\nDelhi?"


def test_db_unreachable_is_503(client, use_engine, unreachable_engine, fake_llm):
    llm = fake_llm(FakeLLM(CITED))
    use_engine(unreachable_engine)
    app.dependency_overrides[query_embedder] = FakeEmbedder  # no real embedding call
    response = post(client, "Do you install in Delhi?")
    assert response.status_code == 503
    assert set(response.json()) == {"detail", "request_id"}
    assert llm.calls == []


def test_db_not_configured_is_503(client, use_engine, fake_llm):
    llm = fake_llm(FakeLLM(CITED))
    use_engine(None)
    app.dependency_overrides[query_embedder] = FakeEmbedder
    assert post(client, "Do you install in Delhi?").status_code == 503
    assert llm.calls == []


def test_end_to_end_with_the_test_database(client, use_engine, fake_llm, test_engine):
    embedder = FakeEmbedder()
    ingest_sections(SECTIONS, "abc123def456", test_engine, embedder)
    embedder.query_vector = mix((WARRANTY, 1.0))  # the question is closest to the warranty chunk
    use_engine(test_engine)
    app.dependency_overrides[query_embedder] = lambda: embedder
    fake_llm(FakeLLM("Panels carry a guarantee [KB Section 11: Warranty and Support]."))

    body = post(client, "Is there a guarantee on the panels?").json()
    assert body["steps"][0]["data"]["mode"] == "hybrid"
    assert body["sources"] == [
        {
            "section": "11",
            "title": "Warranty and Support",
            "page": 4,
            "snippet": "Panels carry a guarantee against defects.",
        }
    ]

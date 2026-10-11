"""GET /api/welcome serves the same text the agent sends for a plain hello."""
from app.agent.graph import get_agent
from app.agent.welcome import STARTER_QUESTIONS, WELCOME_TEXT
from app.main import app
from tests.agent_fakes import Scripted, answer, make_graph, understood


def test_welcome_endpoint(client):
    response = client.get("/api/welcome")
    assert response.status_code == 200
    assert response.json() == {"text": WELCOME_TEXT, "starter_questions": STARTER_QUESTIONS}
    assert 3 <= len(STARTER_QUESTIONS) <= 4


def test_welcome_text_names_the_limits():
    for words in ("exact prices", "savings", "subsidy amounts", "installation dates", "Vantis team confirms"):
        assert words in WELCOME_TEXT


def test_chat_hello_and_endpoint_use_the_same_words(client):
    graph, _ = make_graph(Scripted(understood(["greeting"])), Scripted(answer("unused")))
    app.dependency_overrides[get_agent] = lambda: graph
    reply = client.post("/api/chat", json={"message": "what can you do?"}).json()["reply"]
    assert reply == client.get("/api/welcome").json()["text"]

"""Conversation memory with the real Postgres checkpointer on the Neon test branch (fake models)."""
import uuid

import pytest
from langgraph.checkpoint.postgres import PostgresSaver

from app.agent.memory import CONNECTION_KWARGS, setup_checkpointer
from app.core.config import get_settings
from tests.agent_fakes import Scripted, answer, make_graph, run, understood


@pytest.fixture
def saver(test_engine):  # test_engine skips the test when TEST_DATABASE_URL is not set
    import psycopg

    with psycopg.connect(get_settings().test_database_url, **CONNECTION_KWARGS) as conn:
        checkpointer = PostgresSaver(conn)
        assert setup_checkpointer(checkpointer) is True  # creates the checkpoint tables if missing
        yield checkpointer


def test_history_survives_a_new_graph_instance(saver):
    thread = f"test-{uuid.uuid4()}"
    understand_1 = Scripted(understood(["question"], "Delhi"))
    graph_1, _ = make_graph(understand_1, Scripted(answer("Delhi is listed.")), checkpointer=saver)
    run(graph_1, "Do you provide installation in Delhi?", thread=thread)

    # A new graph on the same database, as after a server restart
    understand_2 = Scripted(understood(["question"], "Kanpur"))
    graph_2, _ = make_graph(understand_2, Scripted(answer("Kanpur too.")), checkpointer=saver)
    state = run(graph_2, "And in Kanpur?", thread=thread)

    prompt = understand_2.calls[0][0].content
    assert "Customer: Do you provide installation in Delhi?" in prompt
    assert "Assistant: Delhi is listed." in prompt
    assert [m.content for m in state["messages"]] == [
        "Do you provide installation in Delhi?", "Delhi is listed.", "And in Kanpur?", "Kanpur too.",
    ]
    saver.delete_thread(thread)
    assert saver.get_tuple({"configurable": {"thread_id": thread}}) is None


def test_setup_failure_never_raises(unreachable_engine):
    class Broken:
        def setup(self):
            raise OSError("no database")

    assert setup_checkpointer(Broken()) is False
    assert setup_checkpointer(None) is False

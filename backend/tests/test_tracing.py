"""LangSmith tracing: off by default, on only with the flag and a key (values copied to os.environ). T13c: every AI
call in app/core/llm.py is its own run, checked with a mocked LangSmith client (no network)."""
import os
import time
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree, tracing_context

from app.core import llm
from app.core.config import Settings
from app.core.tracing import configure_tracing
from tests.test_embedder import FakeClient
from tests.test_llm import GroqError

KEYS = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_HIDE_INPUTS",
        "LANGSMITH_HIDE_OUTPUTS", "LANGSMITH_HIDE_METADATA")


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_tests_never_trace():
    assert os.environ.get("LANGSMITH_TRACING") == "false"  # set by conftest before the app loads


def test_off_without_the_flag_or_the_key(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    assert configure_tracing(settings()) is False
    assert configure_tracing(settings(langsmith_tracing=True)) is False  # no key
    assert configure_tracing(settings(langsmith_api_key="k")) is False  # flag off
    assert "LANGSMITH_API_KEY" not in os.environ


def test_on_copies_the_settings_and_clears_the_sdk_cache(monkeypatch):
    from langsmith import utils as ls_utils

    on = settings(langsmith_tracing=True, langsmith_api_key="test-key", langsmith_project="p",
                  langsmith_hide_inputs=True, langsmith_hide_outputs=True)
    for key in KEYS:
        monkeypatch.setenv(key, "placeholder")  # monkeypatch restores the original values afterwards
    ls_utils.get_env_var("HIDE_INPUTS")  # fill the SDK's cache with the old value
    assert configure_tracing(on) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "test-key"
    assert (os.environ["LANGSMITH_HIDE_INPUTS"], os.environ["LANGSMITH_HIDE_OUTPUTS"]) == ("true", "true")
    assert os.environ["LANGSMITH_HIDE_METADATA"] == "false"
    assert ls_utils.get_env_var("HIDE_INPUTS") == "true"  # the cache was cleared
    ls_utils.get_env_var.cache_clear()


# ---------- T13c: every AI call in app/core/llm.py is its own run ----------

def traced(fn):
    """Run fn() with tracing on and a mocked client; return the runs it created and updated, by name."""
    client = MagicMock(spec=Client)
    with tracing_context(enabled=True, client=client):
        fn()
    time.sleep(0.6)  # a model cut off at its deadline finishes in its worker thread later
    created = {c.kwargs["name"]: c.kwargs for c in client.create_run.call_args_list}
    updated = {c.kwargs["name"]: c.kwargs for c in client.update_run.call_args_list}
    return created, updated


def metadata(run: dict) -> dict:
    return (run.get("extra") or {}).get("metadata", {})


def test_each_model_attempt_is_a_run_with_its_use_deadline_and_failure_reason():
    def slow(_messages):
        time.sleep(0.3)
        return AIMessage("late")

    def groq(_messages):
        return AIMessage("ok", response_metadata={"model_provider": "groq"}, usage_metadata={
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12})

    chain = llm.model_chain([llm.Candidate("gemini:main", RunnableLambda(slow, name="gemini"), "understand", 0.05),
                             llm.Candidate("groq:m", RunnableLambda(groq, name="groq"), "understand", 1)],
                            llm.SkipList())
    created, updated = traced(lambda: chain.invoke([HumanMessage("my bill is 6k")]))
    assert {"model_chain:understand", "understand: gemini:main", "understand: groq:m", "gemini", "groq"} <= set(created)
    first = created["understand: gemini:main"]
    assert metadata(first) == {**metadata(first), "model": "gemini:main", "use": "understand", "deadline_s": 0.05}
    assert first["parent_run_id"] == created["model_chain:understand"]["id"]
    assert created["gemini"]["parent_run_id"] == first["id"]  # the provider's own run nests in the attempt
    assert first["inputs"]["messages"][0].content == "my bill is 6k"  # the prompt, not the model or the config
    assert metadata(updated["understand: gemini:main"])["failure"] == "timeout"
    assert "DeadlineExceeded" in updated["understand: gemini:main"]["error"]
    assert updated["understand: groq:m"]["outputs"]["tokens"] == 12 and not updated["understand: groq:m"]["error"]


def test_a_skipped_model_and_a_providers_error_code_show_in_the_trace():
    skips = llm.SkipList()
    skips.failed("gemini:main", "answer", "429")

    def bad_request(_messages):
        raise GroqError(400, "invalid_request_error", "tool_use_failed")

    chain = llm.model_chain([
        llm.Candidate("gemini:main", RunnableLambda(lambda m: AIMessage("never")), "answer", 1),
        llm.Candidate("groq:m", RunnableLambda(bad_request), "answer", 1),
        llm.Candidate("gemini:fast", RunnableLambda(lambda m: AIMessage("from fast")), "answer", 1),
    ], skips)
    created, updated = traced(lambda: chain.invoke([HumanMessage("hi")]))
    assert updated["answer: gemini:main (skipped)"]["outputs"]["skipped_for_s"] == 60
    assert metadata(updated["answer: groq:m"]) == {**metadata(updated["answer: groq:m"]), "failure": "GroqError",
                                                   "error_status": 400, "error_code": "tool_use_failed"}
    assert not updated["answer: gemini:fast"]["error"]


def test_the_query_embedding_is_a_run_with_its_length_not_the_vector():
    embedder = llm.Embedder(FakeClient(), "fake-model", 4)
    created, updated = traced(lambda: embedder.embed_query("Do you install in Kanpur?"))
    run = updated["embed: gemini-embed:fake-model"]
    assert run["run_type"] == "embedding" and run["outputs"] == {"dimensions": 4}
    assert created["embed: gemini-embed:fake-model"]["inputs"] == {"text": "Do you install in Kanpur?"}


def test_with_tracing_off_there_is_no_run():
    seen = []

    def model(_messages):
        seen.append(get_current_run_tree())
        return AIMessage("ok")

    llm.model_chain([llm.Candidate("gemini:main", RunnableLambda(model), "answer", 1)], llm.SkipList()).invoke(
        [HumanMessage("hi")])
    assert seen == [None]

"""scripts/serve_test_branch.py: it must refuse to run unless leads go to a separate test database, and its
--slow-gemini drill switch (T12a) exists only there."""
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app.core import llm
from app.core.config import REPO_ROOT, Settings
from scripts import serve_test_branch
from scripts.serve_test_branch import engine_for_test_branch

MAIN = "postgresql://u:p@main-host/db?sslmode=require"
TEST = "postgresql://u:p@test-host/db?sslmode=require"


def settings(database_url: str, test_database_url: str) -> Settings:
    # _env_file=None: only the values given here, never backend/.env
    return Settings(_env_file=None, database_url=database_url, test_database_url=test_database_url)


def test_refuses_without_a_test_database():
    with pytest.raises(SystemExit, match="TEST_DATABASE_URL is not set"):
        engine_for_test_branch(settings(MAIN, ""))


def test_refuses_when_the_test_database_is_main():
    with pytest.raises(SystemExit, match="same as DATABASE_URL"):
        engine_for_test_branch(settings(MAIN, MAIN))


def test_refuses_when_only_the_driver_prefix_differs():
    with pytest.raises(SystemExit, match="same as DATABASE_URL"):
        engine_for_test_branch(settings(MAIN, MAIN.replace("postgresql://", "postgres://")))


def test_builds_an_engine_for_the_test_branch_without_connecting():
    engine = engine_for_test_branch(settings(MAIN, TEST))
    assert engine.url.host == "test-host"


# ---------- --slow-gemini: the slow-Google drill (T12a) ----------

def test_slow_gemini_makes_gemini_hang_so_the_deadline_hands_over(monkeypatch):
    monkeypatch.setattr(llm, "gemini_chat", llm.gemini_chat)  # restored after the test
    monkeypatch.setattr(serve_test_branch, "HANG_S", 0.5)  # frees the worker soon; the drill hangs 10 s
    serve_test_branch.force_gemini_timeouts()
    fake_keys = Settings(_env_file=None, google_api_key="fake-key")
    gemini = llm.gemini_chat("gemini-main", fake_keys)
    assert isinstance(gemini, serve_test_branch.HangingGemini)
    groq = RunnableLambda(lambda _m: AIMessage("from groq", response_metadata={"model_provider": "groq"}))
    skips = llm.SkipList()
    chain = llm.model_chain([llm.Candidate("gemini:gemini-main", gemini, "understand", 0.2),
                             llm.Candidate("groq:m", groq, "understand", 1)], skips)
    start = time.perf_counter()
    assert chain.invoke([HumanMessage("hi")]).content == "from groq"
    assert time.perf_counter() - start < 1  # cut off at the deadline, not after the hang
    assert skips.left("gemini:gemini-main") > 0  # later turns go straight to Groq


def test_the_drill_switch_can_never_reach_render():
    assert not any(word in name for name in Settings.model_fields for word in ("slow", "force", "drill"))
    copied = [line for line in (REPO_ROOT / "Dockerfile").read_text().splitlines() if line.startswith("COPY")]
    assert copied and not any("scripts" in line for line in copied)  # backend/scripts is not in the image

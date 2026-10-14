"""Shared fixtures."""
import os

# Tests never send traces to LangSmith, whatever backend/.env says (the environment beats .env).
os.environ["LANGSMITH_TRACING"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

from app.api import chat as chat_api  # noqa: E402
from app.api import security  # noqa: E402
from app.core.config import get_settings, to_sqlalchemy_url  # noqa: E402
from app.core.db import get_engine, init_db  # noqa: E402
from app.core.llm import SKIPS  # noqa: E402
from app.main import app  # noqa: E402

# Nothing listens on port 1, so connecting fails instantly.
UNREACHABLE_URL = "postgresql+psycopg://u:p@127.0.0.1:1/x"
EMPTY_LEAD_TABLES = text("TRUNCATE leads, callbacks, support_requests RESTART IDENTITY CASCADE")


@pytest.fixture(autouse=True)
def fresh_limits():
    """Every test starts with empty rate-limit counters and no saved replies (they are per process)."""
    security.CHAT_LIMITER.reset()
    security.KEY_FAILURES.reset()
    chat_api.RECENT_REPLIES.reset()


@pytest.fixture(autouse=True)
def fresh_skips():
    """Every test starts with no model in a skip window (the skip list is per process, T12a)."""
    SKIPS.clear()
    yield
    SKIPS.clear()


@pytest.fixture
def client():
    """TestClient without `with`, so the startup hook does not touch the real DB."""
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def use_engine():
    """Return a function that makes the app use a given engine (or None)."""

    def _use(engine):
        app.dependency_overrides[get_engine] = lambda: engine

    return _use


@pytest.fixture
def unreachable_engine():
    return create_engine(UNREACHABLE_URL, connect_args={"connect_timeout": 2})


@pytest.fixture
def test_engine():
    """Engine for TEST_DATABASE_URL; skips the test when it is not set.

    Also skips if it points at the same database as DATABASE_URL, so tests never write to it.
    """
    settings = get_settings()
    if not settings.test_database_url:
        pytest.skip("TEST_DATABASE_URL not set")
    if to_sqlalchemy_url(settings.test_database_url) == settings.sqlalchemy_database_url:
        pytest.skip("TEST_DATABASE_URL is the same as DATABASE_URL; refusing to run DB tests")
    url = to_sqlalchemy_url(settings.test_database_url)
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})


@pytest.fixture
def leads_db(test_engine):
    """The test branch with empty lead tables, emptied again afterwards, so the tests never leave leads,
    callbacks or support requests behind on the test branch (T12a; the demo data is seeded after the tests)."""
    init_db(test_engine)
    with test_engine.begin() as conn:
        conn.execute(EMPTY_LEAD_TABLES)
    yield test_engine
    with test_engine.begin() as conn:
        conn.execute(EMPTY_LEAD_TABLES)

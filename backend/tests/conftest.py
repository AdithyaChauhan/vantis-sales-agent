"""Shared fixtures."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.core.config import get_settings, to_sqlalchemy_url
from app.core.db import get_engine
from app.main import app

# Nothing listens on port 1, so connecting fails instantly.
UNREACHABLE_URL = "postgresql+psycopg://u:p@127.0.0.1:1/x"


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
    """Engine for TEST_DATABASE_URL; skips the test when it is not set."""
    settings = get_settings()
    if not settings.test_database_url:
        pytest.skip("TEST_DATABASE_URL not set")
    url = to_sqlalchemy_url(settings.test_database_url)
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})

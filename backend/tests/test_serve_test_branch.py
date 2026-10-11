"""scripts/serve_test_branch.py: it must refuse to run unless leads go to a separate test database."""
import pytest

from app.core.config import Settings
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

"""LangSmith tracing: off by default, on only with the flag and a key (values copied to os.environ)."""
import os

from app.core.config import Settings
from app.core.tracing import configure_tracing

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

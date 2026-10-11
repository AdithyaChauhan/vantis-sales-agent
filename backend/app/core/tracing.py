"""Optional LangSmith tracing. Off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are both set.

pydantic-settings reads backend/.env into Settings but does not put the values into os.environ, while the
LangSmith SDK reads os.environ (and caches what it read). So this copies the LANGSMITH_* settings into the
environment once at startup, before any chain runs, and clears the SDK's cache. Values are never logged.

Hiding (langsmith 0.14.7): LANGSMITH_HIDE_INPUTS, LANGSMITH_HIDE_OUTPUTS and LANGSMITH_HIDE_METADATA hide
that part of every trace when set to exactly "true".
"""
import logging
import os

from app.core.config import Settings

logger = logging.getLogger(__name__)


def configure_tracing(settings: Settings) -> bool:
    """Turn LangSmith tracing on when it is enabled and a key is set. Returns True if it is on."""
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        return False
    values = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": settings.langsmith_api_key,
        "LANGSMITH_PROJECT": settings.langsmith_project,
        "LANGSMITH_HIDE_INPUTS": str(settings.langsmith_hide_inputs).lower(),
        "LANGSMITH_HIDE_OUTPUTS": str(settings.langsmith_hide_outputs).lower(),
        "LANGSMITH_HIDE_METADATA": str(settings.langsmith_hide_metadata).lower(),
    }
    os.environ.update(values)
    try:
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()  # the SDK caches environment reads
    except ImportError:  # pragma: no cover - langsmith comes with langchain-core
        pass
    logger.info(
        "langsmith tracing on",
        extra={"project": settings.langsmith_project, "hide_inputs": settings.langsmith_hide_inputs,
               "hide_outputs": settings.langsmith_hide_outputs},
    )
    return True

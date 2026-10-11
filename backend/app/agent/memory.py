"""Conversation memory: LangGraph's Postgres checkpointer on Neon, one thread per session_id.

The checkpointer creates its own tables (setup()), like init_db does for ours. It stores the full
conversation, including personal details; deleting old threads is future work (PLAN.md).
"""
import logging
from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# What PostgresSaver needs from every connection
CONNECTION_KWARGS = {"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row, "connect_timeout": 10}


@lru_cache
def get_checkpointer() -> PostgresSaver | None:
    """Saver over a small connection pool on DATABASE_URL, or None when it is not set.

    min_size=0 keeps no idle connection open (Neon can sleep); connections idle for 4 minutes are closed,
    and each one is checked before use because Neon drops idle connections.
    """
    url = get_settings().database_url
    if not url:
        return None
    pool = ConnectionPool(
        url,
        min_size=0,
        max_size=4,
        max_idle=240,
        timeout=15,
        check=ConnectionPool.check_connection,
        kwargs=CONNECTION_KWARGS,
        open=False,
    )
    pool.open(wait=False)
    return PostgresSaver(pool)


def setup_checkpointer(saver: PostgresSaver | None) -> bool:
    """Create the checkpoint tables if missing. Logs and returns False on failure (never fails the boot)."""
    if saver is None:
        logger.warning("DATABASE_URL not set; conversation memory disabled")
        return False
    try:
        saver.setup()
        logger.info("checkpointer ready")
        return True
    except Exception as exc:
        logger.error("checkpointer setup failed: %s", type(exc).__name__)
        return False

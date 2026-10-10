"""Neon Postgres connection: engine, health check and startup init."""
import logging
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_engine() -> Engine | None:
    """Shared engine with pre-ping (Neon sleeps). None when DATABASE_URL is unset."""
    url = get_settings().sqlalchemy_database_url
    if not url:
        return None
    return create_engine(
        url, pool_pre_ping=True, pool_recycle=300, connect_args={"connect_timeout": 10}
    )


def check_db(engine: Engine | None) -> bool:
    """True if `SELECT 1` works. Never raises."""
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("db check failed: %s", type(exc).__name__)
        return False


def init_db(engine: Engine | None) -> None:
    """Enable pgvector. Logs and returns on failure so startup never crashes."""
    if engine is None:
        logger.warning("DATABASE_URL not set; skipping db init")
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("db init ok: pgvector extension ready")
    except Exception as exc:
        logger.error("db init failed: %s", type(exc).__name__)

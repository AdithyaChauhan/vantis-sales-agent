"""GET /api/health."""
from fastapi import APIRouter, Depends
from sqlalchemy import Engine

from app.core.db import check_db, get_engine
from app.rag.retriever import count_kb_chunks

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health(engine: Engine | None = Depends(get_engine)) -> dict[str, str | int | None]:
    """Report service, database and KB status. Returns 200 even when the DB is down."""
    db_ok = check_db(engine)
    kb_chunks = count_kb_chunks(engine) if db_ok else None  # None: DB down or table missing
    return {
        "status": "ok" if db_ok else "degraded",
        "db": "ok" if db_ok else "error",
        "kb_chunks": kb_chunks,
    }


@router.get("/api/health/live")
def live() -> dict[str, str]:
    """Render's health check: the process is up. No database query, so it stays fast (Render allows
    5 s) and lets Neon auto-suspend. Use /api/health for the full check."""
    return {"status": "ok"}

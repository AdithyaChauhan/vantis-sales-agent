"""GET /api/health."""
from fastapi import APIRouter, Depends
from sqlalchemy import Engine

from app.core.db import check_db, get_engine

router = APIRouter()


@router.get("/api/health")
def health(engine: Engine | None = Depends(get_engine)) -> dict[str, str]:
    """Report service and database status. Returns 200 even when the DB is down."""
    db_ok = check_db(engine)
    return {"status": "ok" if db_ok else "degraded", "db": "ok" if db_ok else "error"}

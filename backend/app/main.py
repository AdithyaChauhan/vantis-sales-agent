"""FastAPI app: settings, logging, request ids and routers."""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.memory import get_checkpointer, setup_checkpointer
from app.api import chat, health, welcome
from app.core.config import get_settings
from app.core.db import get_engine, init_db
from app.core.logging import request_id_var, setup_logging
from app.core.tracing import configure_tracing

settings = get_settings()
setup_logging(settings.log_level)
configure_tracing(settings)  # optional LangSmith; before anything runs a chain
logger = logging.getLogger("app.request")
# Not logged: Render calls the health check every few seconds and would bury the real logs.
QUIET_PATHS = {"/api/health/live"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: tables and pgvector, then the conversation-memory tables (neither fails the boot)."""
    init_db(get_engine())
    setup_checkpointer(get_checkpointer())
    yield


app = FastAPI(title="Vantis Sales Agent", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Give every request an id, log it, and turn crashes into a JSON 500."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error", extra={"path": request.url.path})
        response = JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong", "request_id": request_id},
        )
    response.headers["X-Request-ID"] = request_id
    if request.url.path in QUIET_PATHS and response.status_code < 400:
        return response
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        },
    )
    return response


# Added after the request-id middleware so CORS is outermost and its headers
# are also present on error responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(welcome.router)

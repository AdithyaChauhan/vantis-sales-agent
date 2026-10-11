"""FastAPI app: settings, logging, request ids, error handlers and routers."""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from app.agent.memory import get_checkpointer, setup_checkpointer
from app.api import chat, health, sales, welcome
from app.api.security import ApiError, error_response
from app.core.config import get_settings
from app.core.db import get_engine, init_db
from app.core.logging import request_id_var, setup_logging
from app.core.tracing import configure_tracing
from app.leads.schemas import error_text

settings = get_settings()
setup_logging(settings.log_level)
configure_tracing(settings)  # optional LangSmith; before anything runs a chain
logger = logging.getLogger("app.request")
# Not logged: Render calls the health check every few seconds and would bury the real logs.
QUIET_PATHS = {"/api/health/live"}
DB_DOWN = "The database is not reachable right now. Please try again in a minute."
DESCRIPTION = """Sales and lead-qualification agent for Vantis Energy Solutions.

**Public:** chat, welcome text and health. `/api/chat` is rate limited per client IP.

**Sales** endpoints need the `X-API-Key` header: click **Authorize**, paste the key, then use *Try it out*.
Errors are JSON with a `detail` message (plus `errors`, `missing_fields` or `retry_after` where useful)."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: tables and pgvector, then the conversation-memory tables (neither fails the boot)."""
    init_db(get_engine())
    setup_checkpointer(get_checkpointer())
    yield


app = FastAPI(title="Vantis Sales Agent", description=DESCRIPTION, lifespan=lifespan)


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
    extra = {
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
    }
    # Where the rate limits got the client IP: "header" (CLIENT_IP_HEADER) or "fallback". Never the IP.
    ip_source = getattr(request.state, "ip_source", None)
    if ip_source:
        extra["ip_source"] = ip_source
    logger.info("request", extra=extra)
    return response


@app.exception_handler(ApiError)
async def api_error(request: Request, exc: ApiError) -> JSONResponse:
    """401, 404, 422, 429 and 503 raised by our code, as {"detail", ..., "request_id"}."""
    return error_response(exc.status_code, exc.detail, **exc.extra)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with one plain message per field. Unlike FastAPI's default, the input is not echoed back."""
    errors: dict[str, str] = {}
    for error in exc.errors():
        loc = [str(part) for part in error["loc"]]
        if loc and loc[0] in ("body", "query", "path", "header"):
            loc = loc[1:]
        field = "body" if error["type"] == "json_invalid" or not loc else ".".join(loc)
        errors.setdefault(field, error_text(error))
    return error_response(422, "Some values are not valid.", errors=errors)


@app.exception_handler(OperationalError)
async def database_down(request: Request, exc: OperationalError) -> JSONResponse:
    """The database is unreachable (Neon waking up or down): 503, never a 500."""
    logger.error("database unavailable: %s", type(exc.orig).__name__, extra={"path": request.url.path})
    return error_response(503, DB_DOWN)


# Added after the request-id middleware so CORS is outermost and its headers are also present on error
# responses. X-API-Key for the dashboard (T11); Retry-After exposed so the chat page can read it (T10).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Retry-After"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(welcome.router)
app.include_router(sales.router)

"""FastAPI app: settings, logging, request ids and routers."""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health
from app.core.config import get_settings
from app.core.db import get_engine, init_db
from app.core.logging import request_id_var, setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("app.request")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: enable pgvector (never fails the boot)."""
    init_db(get_engine())
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

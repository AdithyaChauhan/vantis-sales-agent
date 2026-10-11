"""POST /api/chat: one customer message through the LangGraph agent (app/agent/graph.py).

The conversation is remembered per session_id. `lead` stays null until T8b adds the lead tools.
Never a 500: database trouble is a 503, anything else a friendly degraded reply.
"""
import logging
import time
import unicodedata
import uuid

import psycopg
import psycopg_pool
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import DBAPIError

from app.agent.graph import get_agent
from app.agent.state import turn_input
from app.core.logging import request_id_var
from app.rag.answer import LLM_DOWN_REPLY

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_MESSAGE_CHARS = 1000
DB_DOWN_DETAIL = "Our knowledge base is not reachable right now. Please try again in a minute."
# Search (SQLAlchemy) and memory (psycopg pool) failures: the database is the problem, not the request
DB_ERRORS = (DBAPIError, psycopg.OperationalError, psycopg_pool.PoolTimeout)


def clean_message(text: str) -> str:
    """Remove control characters (keeping newlines and tabs) and trim the ends."""
    kept = (ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    return "".join(kept).strip()


class ChatRequest(BaseModel):
    """A customer message. The conversation is remembered per session_id (a new one when not given)."""

    session_id: str | None = Field(default=None, max_length=100)
    message: str
    channel: str = Field(default="web", max_length=20)

    @field_validator("message")
    @classmethod
    def _clean(cls, value: str) -> str:
        value = clean_message(value)
        if not 1 <= len(value) <= MAX_MESSAGE_CHARS:
            raise ValueError(f"message must be 1-{MAX_MESSAGE_CHARS} characters after trimming")
        return value


class Source(BaseModel):
    """A KB chunk the reply is based on."""

    section: str
    title: str
    page: int
    snippet: str


class Step(BaseModel):
    """One step of the turn, for the agent-steps panel (name is the tool for tool steps)."""

    node: str
    name: str | None = None
    detail: str
    data: dict = {}


class ChatResponse(BaseModel):
    """The reply plus where it came from. degraded = a fallback was used (keyword search, LLM down,
    or the understand step failed). lead stays null until T8b."""

    session_id: str
    reply: str
    sources: list[Source]
    steps: list[Step]
    lead: None = None
    degraded: bool


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent=Depends(get_agent)):
    """Answer one customer message. 503 if the database is down; never a 500."""
    session_id = request.session_id or str(uuid.uuid4())
    if agent is None:  # DATABASE_URL not set
        return _db_down(session_id, "not configured")
    start = time.perf_counter()
    try:
        state = agent.invoke(turn_input(request.message), {"configurable": {"thread_id": session_id}})
    except DB_ERRORS as exc:
        return _db_down(session_id, type(exc).__name__)
    except Exception:
        logger.exception("chat: agent failed", extra={"session_id": session_id})
        return ChatResponse(session_id=session_id, reply=LLM_DOWN_REPLY, sources=[], steps=[], degraded=True)

    steps = state.get("steps", [])
    logger.info(
        "chat",
        extra={
            "session_id": session_id,
            "channel": request.channel,
            "intents": state.get("intents"),
            "mode": state.get("search_mode"),
            "chunks": len(state.get("chunks", [])),
            "nodes": [s["node"] if s["node"] != "tool" else f"tool:{s.get('name')}" for s in steps],
            "providers": [s["data"].get("provider") for s in steps if s["node"] in ("understand", "agent")],
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "degraded": state.get("degraded", False),
        },
    )
    return ChatResponse(
        session_id=session_id,
        reply=state["reply"],
        sources=state.get("sources", []),
        steps=steps,
        degraded=state.get("degraded", False),
    )


def _db_down(session_id: str, cause: str) -> JSONResponse:
    logger.error("chat: database unavailable: %s", cause, extra={"session_id": session_id})
    return JSONResponse(status_code=503, content={"detail": DB_DOWN_DETAIL, "request_id": request_id_var.get()})

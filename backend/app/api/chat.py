"""POST /api/chat: one customer message through the LangGraph agent (app/agent/graph.py).

The conversation is remembered per session_id, a random UUID v4 the server issues on the first message
(a guessable id like "demo-1" would let anyone continue someone else's chat). Rate limited per client IP
(app/api/security.py). Never a 500: database trouble is a 503, anything else a friendly degraded reply.

T13d: the page gives each message a client_message_id and sends it once more when it got no reply at all (a network
error, or a proxy's 502/503/504); a repeated id gets the reply already sent instead of running the turn twice.
"""
import logging
import threading
import time
import unicodedata
import uuid
import zlib
from collections import OrderedDict
from typing import Literal

import psycopg
import psycopg_pool
from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import DBAPIError

from app.agent.graph import get_agent
from app.agent.state import turn_input
from app.api.security import chat_rate_limit
from app.core.config import get_settings
from app.core.llm import turn_scope
from app.core.logging import request_id_var
from app.leads.service import refresh_reason
from app.rag.answer import LLM_DOWN_REPLY

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

MAX_MESSAGE_CHARS = 1000
# Search, memory or a lead write failed (T12b: neutral words, since a lead write is not the knowledge base)
DB_DOWN_DETAIL = "We can't reach our database right now. Please try again in a minute."
# Search (SQLAlchemy) and memory (psycopg pool) failures: the database is the problem, not the request
DB_ERRORS = (DBAPIError, psycopg.OperationalError, psycopg_pool.PoolTimeout)
# One turn at a time per session (no two leads, no clashing memory writes). 64 striped locks keep
# memory bounded; enough for Render's single worker. The UNIQUE session_id on leads guards beyond that.
SESSION_LOCKS = [threading.Lock() for _ in range(64)]
LOCK_TIMEOUT_S = 60
BUSY_REPLY = "I'm still working on your last message. Please wait a moment and try again."
SESSION_ID_ERROR = "session_id must be the id from an earlier reply (a UUID v4); leave it out to start a new chat."
MESSAGE_ID_ERROR = "client_message_id must be a random UUID v4, or left out."
# T13d: the replies kept for a repeated client_message_id (in memory, one worker, like the rate limiter)
RECENT_REPLIES_MAX = 500
RECENT_REPLIES_TTL_S = 600


def session_lock(session_id: str) -> threading.Lock:
    """The lock for a session (crc32 is stable across processes, unlike hash())."""
    return SESSION_LOCKS[zlib.crc32(session_id.encode()) % len(SESSION_LOCKS)]


def uuid4_or_error(value: str, error: str) -> str:
    """The value as a standard UUID v4 string, or ValueError(error) (a nil UUID or another version too)."""
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        raise ValueError(error) from None
    if parsed.version != 4:  # None for the nil UUID and other non-RFC ids
        raise ValueError(error)
    return str(parsed)


class RecentReplies:
    """The last replies by the page's client_message_id (T13d), so a message sent again because no reply arrived gets
    the reply that was already made instead of a second turn (no second lead write, no repeated question). Kept for
    RECENT_REPLIES_TTL_S, at most RECENT_REPLIES_MAX, in memory (one worker). A repeat that names another session gets
    nothing, so an id never returns someone else's reply."""

    def __init__(self, max_items: int = RECENT_REPLIES_MAX, ttl_s: float = RECENT_REPLIES_TTL_S,
                 clock=time.monotonic):
        self._items: OrderedDict[str, tuple[float, "ChatResponse"]] = OrderedDict()
        self._max, self._ttl, self._clock = max_items, ttl_s, clock
        self._lock = threading.Lock()

    def get(self, message_id: str | None, session_id: str | None) -> "ChatResponse | None":
        """The reply saved for this id, if it is recent and (when a session is named) for that session."""
        if not message_id:
            return None
        with self._lock:
            item = self._items.get(message_id)
            if item is None or self._clock() - item[0] > self._ttl:
                return None
            reply = item[1]
        return reply if session_id in (None, reply.session_id) else None

    def put(self, message_id: str | None, reply: "ChatResponse") -> None:
        if not message_id:
            return
        with self._lock:
            self._items[message_id] = (self._clock(), reply)
            self._items.move_to_end(message_id)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def reset(self) -> None:
        """Forget every reply (tests)."""
        with self._lock:
            self._items.clear()


RECENT_REPLIES = RecentReplies()


def clean_message(text: str) -> str:
    """Remove control characters (keeping newlines and tabs) and trim the ends."""
    kept = (ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    return "".join(kept).strip()


class ChatRequest(BaseModel):
    """A customer message. The conversation is remembered per session_id (a new one when not given)."""

    session_id: str | None = Field(
        default=None, max_length=100, description="From an earlier reply; leave it out to start a new chat"
    )
    message: str
    channel: str = Field(default="web", max_length=20)
    client_message_id: str | None = Field(
        default=None, max_length=100,
        description="A random UUID v4 the page gives each message; sent again with the same message when no reply "
                    "arrived, it gets the reply already made instead of a second turn",
    )

    @field_validator("session_id")
    @classmethod
    def _session_id(cls, value: str | None) -> str | None:
        """Only random ids: a UUID v4 (122 random bits), returned in its standard form."""
        return None if value is None else uuid4_or_error(value, SESSION_ID_ERROR)

    @field_validator("client_message_id")
    @classmethod
    def _message_id(cls, value: str | None) -> str | None:
        return None if value is None else uuid4_or_error(value, MESSAGE_ID_ERROR)

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


class LeadSummary(BaseModel):
    """The lead so far, for the chat page's demo panel. status and score are null unless
    CHAT_SHOW_LEAD_SCORE is on (this endpoint is customer-facing). known_fields holds field names only,
    never their values (no phone, email or money amounts)."""

    id: int | None = None
    status: str | None = None
    score: int | None = None
    missing_fields: list[str] = []
    callback_status: str | None = None
    name: str | None = None
    city: str | None = None
    property_type: str | None = None
    customer_type: str | None = None
    known_fields: list[str] = []


AnsweredBy = Literal["model", "knowledge_base", "tools", "fixed"]


class ChatResponse(BaseModel):
    """The reply plus where it came from. degraded = a fallback was used (keyword search, LLM down,
    or the understand step failed). lead is null until something about a lead is known. answered_by (T13d): who wrote
    the reply: a model (maybe a backup one), the knowledge base directly (no model answered: the safety net), the
    tools' outcome in fixed words, or a fixed text."""

    session_id: str
    reply: str
    sources: list[Source]
    steps: list[Step]
    lead: LeadSummary | None = None
    degraded: bool
    answered_by: AnsweredBy = "model"


@router.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(chat_rate_limit)])
def chat(request: ChatRequest, background: BackgroundTasks, agent=Depends(get_agent)):
    """Answer one customer message. 429 over the rate limit, 503 if the database is down; never a 500.

    A lead reason the turn left pending is written by the LLM after the reply (background task).
    """
    session_id = request.session_id or str(uuid.uuid4())
    if agent is None:  # DATABASE_URL not set
        return _db_down(session_id, "not configured")
    repeat = RECENT_REPLIES.get(request.client_message_id, request.session_id)
    if repeat is not None:  # T13d: sent again because no reply arrived: the reply already made
        logger.info("chat: repeated message id, saved reply sent", extra={"session_id": repeat.session_id})
        return repeat
    start = time.perf_counter()
    lock = session_lock(session_id)
    if not lock.acquire(timeout=LOCK_TIMEOUT_S):
        logger.warning("chat: session busy", extra={"session_id": session_id})
        return ChatResponse(session_id=session_id, reply=BUSY_REPLY, sources=[], steps=[], degraded=True,
                            answered_by="fixed")
    repeat = RECENT_REPLIES.get(request.client_message_id, request.session_id)
    if repeat is not None:  # the first send finished while this one waited for the session
        lock.release()
        logger.info("chat: repeated message id, saved reply sent", extra={"session_id": repeat.session_id})
        return repeat
    try:
        with turn_scope():  # T13d: every model call of this turn fits in about 40 s
            state = agent.invoke(
                turn_input(request.message, session_id), {"configurable": {"thread_id": session_id}}
            )
    except DB_ERRORS as exc:
        return _db_down(session_id, type(exc).__name__)
    except Exception:
        logger.exception("chat: agent failed", extra={"session_id": session_id})
        return ChatResponse(session_id=session_id, reply=LLM_DOWN_REPLY, sources=[], steps=[], degraded=True,
                            answered_by="fixed")
    finally:
        lock.release()

    ops = getattr(agent, "lead_ops", None)
    if state.get("reason_pending") and ops is not None:
        background.add_task(refresh_reason, ops.engine, state["reason_pending"], ops.reason_llm)
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
            "answered_by": state.get("answered_by") or "model",
        },
    )
    response = ChatResponse(
        session_id=session_id,
        reply=state["reply"],
        sources=state.get("sources", []),
        steps=steps,
        lead=_lead_summary(state.get("lead_summary")),
        degraded=state.get("degraded", False),
        answered_by=state.get("answered_by") or "model",
    )
    RECENT_REPLIES.put(request.client_message_id, response)
    return response


def _lead_summary(summary: dict | None) -> LeadSummary | None:
    """The demo-panel summary; status and score only when CHAT_SHOW_LEAD_SCORE is on."""
    if not summary:
        return None
    if not get_settings().chat_show_lead_score:
        summary = {**summary, "status": None, "score": None}
    return LeadSummary(**summary)


def _db_down(session_id: str, cause: str) -> JSONResponse:
    logger.error("chat: database unavailable: %s", cause, extra={"session_id": session_id})
    return JSONResponse(status_code=503, content={"detail": DB_DOWN_DETAIL, "request_id": request_id_var.get()})

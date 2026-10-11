"""POST /api/chat (v0): search the KB, answer from the retrieved chunks, return sources and steps.

No agent, tools, leads or memory yet (T7-T8): `lead` is always null and nothing is remembered.
"""
import logging
import unicodedata
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Engine

from app.core.db import get_engine
from app.core.llm import main_llm, query_embedder
from app.core.logging import request_id_var
from app.rag.answer import LLM_DOWN_REPLY, Answer, answer
from app.rag.retriever import search

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_MESSAGE_CHARS = 1000
CONTEXT_CHUNKS = 4
DB_DOWN_DETAIL = "Our knowledge base is not reachable right now. Please try again in a minute."


def clean_message(text: str) -> str:
    """Remove control characters (keeping newlines and tabs) and trim the ends."""
    kept = (ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    return "".join(kept).strip()


class ChatRequest(BaseModel):
    """A customer message. session_id is echoed back (memory comes in T8)."""

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
    """One step of the turn, for the agent-steps panel."""

    node: str
    detail: str
    data: dict = {}


class ChatResponse(BaseModel):
    """The reply plus where it came from. degraded = a fallback was used (keyword search or LLM down)."""

    session_id: str
    reply: str
    sources: list[Source]
    steps: list[Step]
    lead: None = None
    degraded: bool


@router.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    engine: Engine | None = Depends(get_engine),
    embedder=Depends(query_embedder),
    llm: Runnable = Depends(main_llm),
):
    """Answer one customer message from the KB. 503 if the database is down; never a 500."""
    session_id = request.session_id or str(uuid.uuid4())
    try:
        if engine is None:  # search() would quietly use the default engine instead
            raise RuntimeError("DATABASE_URL is not set")
        result = search(request.message, k=CONTEXT_CHUNKS, engine=engine, embedder=embedder)
    except Exception as exc:
        logger.error("chat: KB search failed: %s", type(exc).__name__, extra={"session_id": session_id})
        return JSONResponse(
            status_code=503, content={"detail": DB_DOWN_DETAIL, "request_id": request_id_var.get()}
        )

    try:
        result_answer = answer(request.message, result, llm)
    except Exception:
        logger.exception("chat: answering failed", extra={"session_id": session_id})
        result_answer = Answer(LLM_DOWN_REPLY, [], [], degraded=True)

    answer_data = result_answer.steps[-1]["data"] if result_answer.steps else {}
    logger.info(
        "chat",
        extra={
            "session_id": session_id,
            "channel": request.channel,
            "mode": result.mode,
            "chunks": len(result.chunks),
            "no_context": result.no_context,
            "provider": answer_data.get("provider"),
            "latency_ms": answer_data.get("latency_ms"),
            "degraded": result_answer.degraded,
        },
    )
    return ChatResponse(
        session_id=session_id,
        reply=result_answer.reply,
        sources=result_answer.sources,
        steps=result_answer.steps,
        degraded=result_answer.degraded,
    )

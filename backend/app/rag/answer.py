"""Grounded answer: retrieved KB chunks -> main LLM -> reply with citations, sources and steps.

No FastAPI and no database here: answer() takes a finished SearchResult, so it is easy to test.
"""
import logging
import re
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from app.core.llm import LLMError, invoke_llm
from app.core.prompts import load_prompt
from app.rag.retriever import RetrievedChunk, SearchResult

logger = logging.getLogger(__name__)

NO_CONTEXT_REPLY = (
    "I don't have that information in my Vantis knowledge base. "
    "I can connect you with our sales team if you'd like."
)
LLM_DOWN_REPLY = "I'm having trouble right now. Please try again in a minute."
SNIPPET_CHARS = 160
_CITATION_RE = re.compile(r"KB Section (\d+)")


@dataclass
class Answer:
    """What the chat endpoint sends back (minus session id and lead)."""

    reply: str
    sources: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    degraded: bool = False


def citation_label(chunk: RetrievedChunk) -> str:
    """The label the model must cite, e.g. [KB Section 8: Product & Service Catalog]."""
    return f"[KB Section {chunk.section_no}: {chunk.section_title}]"


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Each chunk as its label on one line and its text below, separated by blank lines."""
    return "\n\n".join(f"{citation_label(c)}\n{c.content}" for c in chunks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[BaseMessage]:
    """System prompt + the RAG prompt filled with the context and the question.

    The context is filled in first, so a "{context}" typed by the customer is left as it is.
    """
    user = load_prompt("rag_prompt").replace("{context}", format_context(chunks))
    user = user.replace("{question}", question)
    return [SystemMessage(load_prompt("system_prompt")), HumanMessage(user)]


def normalize_citations(reply: str) -> str:
    """Turn full-width brackets 【 】 into [ ] (gpt-oss on Groq sometimes cites with them)."""
    return reply.replace("【", "[").replace("】", "]")


def cited_sections(reply: str) -> list[int]:
    """KB Section numbers the reply cites, in order of first mention, without repeats."""
    return list(dict.fromkeys(int(n) for n in _CITATION_RE.findall(reply)))


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The chunk text on one line, cut at a word boundary after at most `limit` characters."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"


def pick_sources(reply: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """One source per KB Section, because citations are per section.

    Uses the context chunks whose section the reply cites (all of them if it cites none of them),
    in retrieval order. A section's first (highest-ranked) chunk gives the snippet.
    """
    cited = set(cited_sections(reply))
    used = [c for c in chunks if c.section_no in cited] or chunks
    if cited - {c.section_no for c in chunks}:
        logger.warning("reply cites a section that was not in the context")
    first_per_section = {}
    for c in used:
        first_per_section.setdefault(c.section_no, c)
    return [
        {
            "section": str(c.section_no),
            "title": c.section_title,
            "page": c.page,
            "snippet": snippet(c.content),
        }
        for c in first_per_section.values()
    ]


def retrieve_step(result: SearchResult) -> dict:
    """The retrieve entry of `steps`."""
    detail = f"{len(result.chunks)} chunks ({result.mode})"
    if result.no_context:
        detail = f"no relevant KB context ({result.mode})"
    return {
        "node": "retrieve",
        "detail": detail,
        "data": {"mode": result.mode, "chunks": len(result.chunks), "no_context": result.no_context},
    }


def answer(question: str, result: SearchResult, llm: Runnable) -> Answer:
    """Answer from the retrieved chunks. Never raises for LLM failures (degraded reply instead).

    No relevant context: a fixed reply, and the LLM is not called.
    """
    steps = [retrieve_step(result)]
    if result.no_context:
        steps.append(
            {
                "node": "answer",
                "detail": "fixed reply, LLM not called",
                "data": {"provider": None, "latency_ms": 0},
            }
        )
        return Answer(NO_CONTEXT_REPLY, [], steps, degraded=result.degraded)

    try:
        llm_reply = invoke_llm(llm, build_messages(question, result.chunks))
    except LLMError:
        steps.append(
            {"node": "answer", "detail": "LLM failed", "data": {"provider": None, "latency_ms": None}}
        )
        return Answer(LLM_DOWN_REPLY, [], steps, degraded=True)

    steps.append(
        {
            "node": "answer",
            "detail": f"{llm_reply.provider}, {llm_reply.latency_ms} ms",
            "data": {"provider": llm_reply.provider, "latency_ms": llm_reply.latency_ms},
        }
    )
    reply = normalize_citations(llm_reply.text)
    return Answer(
        reply,
        pick_sources(reply, result.chunks),
        steps,
        degraded=result.degraded,
    )

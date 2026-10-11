"""Helpers for grounded answers: the context block, citations, sources and the retrieve step.

The agent (app/agent/graph.py) uses them; nothing here calls an LLM or the database.
"""
import logging
import re

from langchain_core.messages import HumanMessage

from app.core.prompts import load_prompt
from app.rag.retriever import RetrievedChunk, SearchResult

logger = logging.getLogger(__name__)

LLM_DOWN_REPLY = "I'm having trouble right now. Please try again in a minute."
SNIPPET_CHARS = 160
_CITATION_RE = re.compile(r"KB Section (\d+)")


def citation_label(chunk: RetrievedChunk) -> str:
    """The label the model must cite, e.g. [KB Section 8: Product & Service Catalog]."""
    return f"[KB Section {chunk.section_no}: {chunk.section_title}]"


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Each chunk as its label on one line and its text below, separated by blank lines."""
    return "\n\n".join(f"{citation_label(c)}\n{c.content}" for c in chunks)


def rag_message(question: str, chunks: list[RetrievedChunk], lead_progress: str = "") -> HumanMessage:
    """The customer's message wrapped in the RAG prompt with the context (empty when nothing was found)
    and the lead progress block (what is known and still needed; "(not tracked)" without the lead tools).

    The customer's text is filled in last, so a "{context}" typed by the customer is left as it is.
    """
    text = load_prompt("rag_prompt").replace("{context}", format_context(chunks))
    text = text.replace("{lead_progress}", lead_progress or "(not tracked)")
    return HumanMessage(text.replace("{question}", question))


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

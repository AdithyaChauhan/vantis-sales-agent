"""The understand step: the fast LLM reads the new message (with recent history) and returns intents,
a standalone search query in the KB's own words, and the lead details the customer stated.

The KB section titles and the KB Section 8 summary chunk come from the database (never typed into code),
so "solar systems" can map to the Product & Service Catalog and "HomePro 5" to 5 kW.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Literal, get_args, get_origin

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, create_model
from sqlalchemy import Engine, text

from app.core.llm import LLMError, provider_of
from app.core.prompts import load_prompt
from app.leads.schemas import LeadData
from app.rag.chunker import CATALOG_SECTION

logger = logging.getLogger(__name__)

HISTORY_MESSAGES = 6
HISTORY_CHARS = 500  # each earlier message is cut to this length in the prompt

Intent = Literal[
    "question",
    "buying_interest",
    "greeting",
    "reply_to_assistant",
    "thanks_or_bye",
    "existing_customer_issue",
    "off_topic",
]


def _loose(annotation):
    """Keep enum and bool types (the LLM should pick valid values); everything else as raw text or number,
    so that a bad phone number becomes a validation error later instead of breaking the whole output."""
    args = get_args(annotation)
    if bool in args or any(get_origin(a) is Literal for a in args):
        return annotation
    return str | int | float | None


# Same field names and descriptions as LeadData (one source of truth), with loose types
ExtractedDetails = create_model(
    "ExtractedDetails",
    **{
        name: (_loose(info.annotation), Field(None, description=info.description))
        for name, info in LeadData.model_fields.items()
    },
)


class Understanding(BaseModel):
    """What the fast LLM returns for one customer message."""

    intents: list[Intent] = Field(description="Every intent present in the new message")
    search_query: str | None = Field(
        None,
        description="A standalone search query in the KB's own words for what the customer asks about "
        "Vantis (earlier messages resolved); null when nothing about Vantis is asked",
    )
    details: ExtractedDetails = Field(  # type: ignore[valid-type]
        default_factory=ExtractedDetails,
        description="Only details the customer stated about themselves in the conversation",
    )


@dataclass
class KbHints:
    """KB section titles and the table summary chunks, read from kb_chunks."""

    titles: str = ""
    summaries: dict[int, dict] = field(default_factory=dict)  # section_no -> chunk dict

    @property
    def catalog(self) -> str:
        chunk = self.summaries.get(CATALOG_SECTION)
        if not chunk:
            return ""
        return f"[KB Section {chunk['section_no']}: {chunk['section_title']}]\n{chunk['content']}"


_TITLES_SQL = text(
    "SELECT DISTINCT section_no, section_title FROM kb_chunks WHERE chunk_type <> 'policy' ORDER BY section_no"
)
_SUMMARIES_SQL = text(
    "SELECT chunk_id, section_no, section_title, page, chunk_type, content FROM kb_chunks "
    "WHERE chunk_type = 'table_summary' ORDER BY section_no"
)
_hints_cache: dict[int, KbHints] = {}


def load_kb_hints(engine: Engine | None) -> KbHints:
    """Titles and summaries from the DB, cached per process once read. Empty (not cached) if the DB fails."""
    if engine is None:
        return KbHints()
    key = id(engine)
    if key in _hints_cache:
        return _hints_cache[key]
    try:
        with engine.connect() as conn:
            titles = [f"KB Section {r.section_no}: {r.section_title}" for r in conn.execute(_TITLES_SQL)]
            summaries = {r.section_no: dict(r._mapping) for r in conn.execute(_SUMMARIES_SQL)}
    except Exception as exc:
        logger.warning("kb hints unavailable: %s", type(exc).__name__)
        return KbHints()
    hints = KbHints("\n".join(titles), summaries)
    _hints_cache[key] = hints
    return hints


def format_history(history: list[BaseMessage]) -> str:
    """Earlier messages as 'Customer: …' / 'Assistant: …' lines, each cut to HISTORY_CHARS."""
    lines = []
    for message in history[-HISTORY_MESSAGES:]:
        who = "Assistant" if isinstance(message, AIMessage) else "Customer"
        content = " ".join(str(message.content).split())
        lines.append(f"{who}: {content[:HISTORY_CHARS]}")
    return "\n".join(lines) or "(no earlier messages)"


def build_prompt(hints: KbHints, message: str, history: list[BaseMessage]) -> str:
    """The understand prompt with the KB hints, the history and the new message filled in."""
    filled = {
        "{section_titles}": hints.titles or "(not available)",
        "{catalog}": hints.catalog or "(not available)",
        "{history}": format_history(history),
    }
    prompt = load_prompt("lead_extraction_prompt")
    for placeholder, value in filled.items():
        prompt = prompt.replace(placeholder, value)
    return prompt.replace("{message}", message)  # last, so braces in the message are left alone


def understand(llm: Runnable, hints: KbHints, message: str, history: list[BaseMessage]) -> dict:
    """{intents, search_query, details (raw), provider, latency_ms}. Raises LLMError on any failure."""
    start = time.perf_counter()
    try:
        result = llm.invoke([HumanMessage(build_prompt(hints, message, history))])
    except Exception as exc:
        raise LLMError(f"understand failed: {type(exc).__name__}") from exc
    latency_ms = round((time.perf_counter() - start) * 1000)
    parsed = result.get("parsed") if isinstance(result, dict) else None
    if parsed is None:
        raise LLMError("understand returned no parsable output")
    raw = result.get("raw")
    query = (parsed.search_query or "").strip() or None
    return {
        "intents": list(dict.fromkeys(parsed.intents)),
        "search_query": query,
        "details": parsed.details.model_dump(exclude_none=True),
        "provider": provider_of(raw) if raw is not None else "unknown",
        "latency_ms": latency_ms,
    }

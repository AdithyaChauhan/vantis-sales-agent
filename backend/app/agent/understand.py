"""The understand step: an LLM (app: understand_llm, i.e. main, then fast, 5 s each) reads the new
message (with recent history) and returns intents, a standalone search query in the KB's own words, and the
lead details the customer stated.

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

from app.core.llm import LLMError, provider_of, tokens_of
from app.core.prompts import load_prompt
from app.leads.schemas import LeadData
from app.rag.chunker import CATALOG_SECTION, PACKAGE_GUIDANCE_SECTION

logger = logging.getLogger(__name__)

HISTORY_MESSAGES = 6
HISTORY_CHARS = 500  # each earlier message is cut to this length in the prompt
# T13e: the assistant's last message goes in its own block, its last this many characters (where its question is): the
# history keeps only each message's first 500, so a long reply lost the question the new message answers
LAST_REPLY_CHARS = 1500

Intent = Literal[
    "question",
    "buying_interest",
    "greeting",
    "reply_to_assistant",
    "contact_request",
    "gives_details",
    "asks_what_is_needed",
    "thanks_or_bye",
    "existing_customer_issue",
    "off_topic",
]
# T13e: what the new message does with the assistant's last message (the AI understands, the code enforces: the graph
# routes, counts skipped questions and parks asks and offers from this field)
ReplyType = Literal["answers", "skips", "accepts", "declines", "goes_ahead", "closes", "asks_new"]
# T13e: how the customer's place relates to the city it names; code confirms only a place inside a listed city
PlaceRelation = Literal["inside", "near", "region", "unsure", "none"]
# The lead details a customer can rule out (the names app/leads/scoring.py missing_details uses)
RuledOut = Literal["timeline", "monthly bill or usage", "rooftop availability", "ownership", "email", "phone number"]
# T13e: whether the customer's installed system has a problem right now; only "yes" starts the support flow
ProblemNow = Literal["yes", "no", "what_if", "unsure"]


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
    """What the understand model returns for one customer message."""

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
    asked_city: str | None = Field(
        None, description="A city the customer only asked about (not stated as theirs), as written"
    )
    cities: list[str] = Field(
        default_factory=list,
        description="Every city or town named in the new message, as written (their own or only asked about)",
    )
    callback_time: str | None = Field(
        None, description="When the customer wants to be called back, copied word for word"
    )
    issue: str | None = Field(
        None, description="An existing customer's problem with their system, in the customer's words"
    )
    project_details: str | None = Field(
        None, description="An existing customer's project details (system, install date, city), as written"
    )
    declines: bool = Field(
        False, description="True if the message says no to, or refuses, what the assistant last asked"
    )
    reply_type: ReplyType = Field(
        description="What the new message does with the assistant's last message (see the rules)"
    )
    ruled_out: list[RuledOut] = Field(
        default_factory=list, description="Their own details the customer excludes or won't share"
    )
    locality: str | None = Field(
        None, description="Anything finer than the city in the place named (a sector, colony or village), as written"
    )
    place_relation: PlaceRelation = Field(
        description="How the place in details.city or asked_city relates to that city; none when no place is named"
    )
    asks_offerings: bool = Field(
        description="True when the customer asks what Vantis offers or provides in general"
    )
    problem_now: ProblemNow = Field(
        description="Whether the customer's installed system has a problem right now (see the rules)"
    )


@dataclass
class KbHints:
    """KB section titles, the table summary chunks and (T13a) the package-guidance rows, read from kb_chunks."""

    titles: str = ""
    summaries: dict[int, dict] = field(default_factory=dict)  # section_no -> chunk dict
    guidance: list[dict] = field(default_factory=list)  # KB Section 9 rows: which reference package fits which need

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
_GUIDANCE_SQL = text(
    "SELECT chunk_id, section_no, section_title, page, chunk_type, content FROM kb_chunks "
    "WHERE section_no = :section AND chunk_type = 'table_row' ORDER BY chunk_id"
)
_hints_cache: dict[int, KbHints] = {}


def load_kb_hints(engine: Engine | None) -> KbHints:
    """Titles, summaries and the package guidance from the DB, cached per process once read. Empty (not cached) if
    the DB fails."""
    if engine is None:
        return KbHints()
    key = id(engine)
    if key in _hints_cache:
        return _hints_cache[key]
    try:
        with engine.connect() as conn:
            titles = [f"KB Section {r.section_no}: {r.section_title}" for r in conn.execute(_TITLES_SQL)]
            summaries = {r.section_no: dict(r._mapping) for r in conn.execute(_SUMMARIES_SQL)}
            guidance = [dict(r._mapping) for r in conn.execute(_GUIDANCE_SQL, {"section": PACKAGE_GUIDANCE_SECTION})]
    except Exception as exc:
        logger.warning("kb hints unavailable: %s", type(exc).__name__)
        return KbHints()
    hints = KbHints("\n".join(titles), summaries, guidance)
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


def last_reply(history: list[BaseMessage]) -> str:
    """The assistant's last message, whole up to its last LAST_REPLY_CHARS characters (T13e), or "(none)"."""
    previous = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)
    if previous is None:
        return "(none)"
    return " ".join(str(previous.content).split())[-LAST_REPLY_CHARS:]


def build_prompt(hints: KbHints, message: str, history: list[BaseMessage]) -> str:
    """The understand prompt with the KB hints, the history, the assistant's last message and the new message."""
    filled = {
        "{section_titles}": hints.titles or "(not available)",
        "{catalog}": hints.catalog or "(not available)",
        "{history}": format_history(history),
        "{last_reply}": last_reply(history),
    }
    prompt = load_prompt("lead_extraction_prompt")
    for placeholder, value in filled.items():
        prompt = prompt.replace(placeholder, value)
    return prompt.replace("{message}", message)  # last, so braces in the message are left alone


def understand(llm: Runnable, hints: KbHints, message: str, history: list[BaseMessage]) -> dict:
    """{intents, search_query, details (raw), asked_city, cities, callback_time, issue, project_details, declines,
    reply_type, ruled_out, locality, place_relation, asks_offerings, problem_now, provider, model, latency_ms,
    tokens}. Raises LLMError on any failure."""
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

    def text(value: str | None) -> str | None:
        return " ".join(value.split()) or None if value else None

    return {
        "intents": list(dict.fromkeys(parsed.intents)),
        "search_query": text(parsed.search_query),
        "details": parsed.details.model_dump(exclude_none=True),
        "asked_city": text(parsed.asked_city),
        "cities": [name for name in (text(c) for c in parsed.cities) if name],
        "callback_time": text(parsed.callback_time),
        "issue": text(parsed.issue),
        "project_details": text(parsed.project_details),
        "declines": bool(parsed.declines),
        "reply_type": parsed.reply_type,
        "ruled_out": list(dict.fromkeys(parsed.ruled_out)),
        "locality": text(parsed.locality),
        "place_relation": parsed.place_relation,
        "asks_offerings": bool(parsed.asks_offerings),
        "problem_now": parsed.problem_now,
        "provider": provider_of(raw) if raw is not None else "unknown",
        "model": raw.response_metadata.get("model_name") if raw is not None else None,
        "latency_ms": latency_ms,
        "tokens": tokens_of(result),
    }

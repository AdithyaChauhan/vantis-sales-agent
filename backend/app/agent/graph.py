"""The LangGraph agent for one customer message (PLAN.md part 3.1).

understand ─┬─ only a greeting ─► welcome ─┐
            ├─ only thanks/bye ─► closing ─┤
            ├─ only off-topic, no query ─► redirect ─┤
            ├─ search_query (not details-only) ─► retrieve ─► agent ⇄ tools ─┤
            └─ anything else ─────────────────────────────► agent ⇄ tools ─┴─► finalize

Every external piece (LLMs, search, service areas, KB hints, leads database, checkpointer) comes in
through AgentDeps, so tests run the real graph with fakes. The lead tools (T8b) read the draft from the
state; finalize saves new details of an existing lead once (if the update_lead tool did not already).

T13d: when every answer model failed (and the last try too), the agent node only marks the turn; finalize then saves a
ready lead the model didn't save and answers from this turn's KB chunks (app/agent/fallback.py), never "try again".
"""
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent import welcome
from app.agent.fallback import fallback_reply
from app.agent.leads import (
    LeadOps, add_saved_line, changed_fields, chat_intent, confirmed_city, drop_detail_questions,
    drop_known_city_question, fix_callback_claims, is_clock_time, is_explorer, keep_one_question, lead_summary,
    merge_callback_time, package_case, pending_changes, place_type, remove_false_claims, resolve_city,
    ruled_out_details, save_ready_lead, short_answer_details, skipped_asks, status_block,
)
from app.agent.packages import (
    SIZE_RE, add_all_packages, add_offerings_line, add_size_line, all_or_no_packages, reference_packages,
)
from app.agent.questions import ask_rules, asked_contact, fix_region_reply, made_offer, offers_made, previous_asked
from app.agent.state import AgentState
from app.agent.text_details import (
    asks_to_be_called, asks_what_is_offered, details_from_text, reply_type_from_words, short_reply,
)
from app.agent.tools import make_tools, service_area_data, tool_step, unverified_area_data
from app.agent.understand import KbHints, Understanding, load_kb_hints, understand
from app.core.config import get_settings
from app.core.db import get_engine
from app.core.llm import LLMError, chat_llm, provider_of, query_embedder, reply_text, tokens_of, understand_llm
from app.leads import service
from app.core.prompts import load_prompt
from app.leads.schemas import LeadData
from app.rag.answer import (
    add_missing_tag, citation_label, normalize_citations, pick_sources, plain_kb_words, plain_service_area_words,
    rag_message, remove_free_claims, remove_kb_rules, remove_uncovered_policies, retrieve_step, soften_promises,
    strip_tool_citations, strip_unsourced_citations, tidy_citations,
)
from app.rag.chunker import CATALOG_SECTION, SERVICE_AREA_SECTION, SUMMARY_SECTIONS
from app.rag.numbers import remove_unsupported_numbers
from app.rag.retriever import RetrievedChunk, SearchResult, search
from app.services.service_areas import ServiceAreas, get_service_areas

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
AGENT_HISTORY = 12  # persisted messages the agent sees, including the current one
CONTEXT_CHUNKS = 4
# No KB search for a message that only gives details, asks to be contacted or asks what details are still
# needed (a short "yes please" may come with it); a short reply alone that accepts an info offer still
# searches. T12a: with no KB context, "is rooftop info not needed?" can't be answered with a KB quote.
NO_SEARCH_INTENTS = {"gives_details", "contact_request", "asks_what_is_needed", "reply_to_assistant"}
NO_SEARCH_TRIGGERS = {"gives_details", "contact_request", "asks_what_is_needed"}
# T12b: every city named in the message is checked in code before the model answers (it saves one model round per
# city turn, which Groq's 8K tokens a minute could not carry in the slow-Google drill); not for these intents alone
MAX_CITY_CHECKS = 3
NO_CITY_CHECK_INTENTS = {"greeting", "thanks_or_bye", "off_topic"}


@dataclass
class AgentDeps:
    """Everything the graph needs from outside."""

    understand_llm: Runnable  # structured output (Understanding, include_raw); app: understand_llm()
    agent_llm: Runnable  # main model with the tools bound
    final_llm: Runnable  # main model without tools (used after MAX_TOOL_ROUNDS)
    search_fn: Callable[[str], SearchResult]
    areas_fn: Callable[[], ServiceAreas]
    hints_fn: Callable[[], KbHints]
    tools: list[BaseTool]
    checkpointer: BaseCheckpointSaver | None = None
    ops: LeadOps | None = None  # the leads database for the lead tools; None: no lead tools (T8a)
    # T13c: every tool but create_lead, for the rest of a turn in which create_lead said details are missing (chat 1
    # called it twice in one turn, "missing: city" both times); None: agent_llm stays
    agent_llm_without_create: Runnable | None = None
    # Off: in the T8a eval it was not needed for q01 and added the whole catalog to 5 unrelated
    # questions (docs/EVAL.md). Kept switchable so the eval keeps measuring it.
    small_to_big: bool = False


def add_table_summaries(chunks: list[dict], summaries: dict[int, dict]) -> list[dict]:
    """Small-to-big: when rows of KB Section 8 or 10 were found but not the table's summary, add it,
    so an answer can list every package (or every service region), not just the rows that matched."""
    present = {c["chunk_id"] for c in chunks}
    added = []
    for section in sorted(SUMMARY_SECTIONS):
        summary = summaries.get(section)
        has_rows = any(c["section_no"] == section and c["chunk_type"] == "table_row" for c in chunks)
        if summary and has_rows and summary["chunk_id"] not in present:
            added.append(summary)
    return chunks + added


def _norm_words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower().replace("&", " and ")).split())


def names_catalog(query: str | None, summary: dict | None) -> bool:
    """True when the understand step's search query is about the KB Section 8 catalog: it names the section's title or
    one of its offerings (KB data). The whole list then joins the context, so a "what do you offer" or a 5 kW answer can
    name every package (T13d; T13e: read from the AI's query, not from the customer's words)."""
    if not query or not summary:
        return False
    padded = f" {_norm_words(query)} "
    rows = [line[2:].split(";")[0] for line in str(summary["content"]).split("\n") if line.startswith("- ")]
    names = [_norm_words(summary["section_title"])] + [_norm_words(row) for row in rows]
    names += [name.removeprefix("vantis ") for name in names]
    return any(name and f" {name} " in padded for name in names)


def place_rules(out: dict, areas: ServiceAreas | None, message: str = "") -> tuple[dict, list[str], set[str]]:
    """The place a message names, as the understand step split it (T13e): the city alone by its official name, a finer
    locality, and how the place relates to the city (place_relation). Code confirms only a place inside a listed city:
    - a city the message uses only inside a KB region's name is that region (ServiceAreas.region_around: KB names
      only, so it can stop a confirmation, never make one; in the live test the fallback model wrote "Delhi" for
      "Delhi NCR");
    - near: the place becomes "<locality, or 'the area'> near <city>", checked as unverified and saved so;
    - unsure, or a place with no relation given: checked as unverified and never stored, so the city is asked again;
    - a KB region in any field ("Delhi NCR"): checked as a region and never stored as the city; a place called a
      region that is no KB region (or is a city) is unverified and never stored.
    Returns (out, the places checked as unverified without a lookup, the lowercased names never stored as the city)."""
    unverified: list[str] = []
    not_stored: set[str] = set()

    def is_region(name: str) -> bool:
        try:
            return areas is not None and areas.lookup(name).status == "region"
        except Exception:  # the list can't be read: nothing is a region
            return False

    def regional(name):  # only ever turns a listed city into a region, never the other way
        try:
            if areas is None or not message or not name or areas.lookup(str(name)).status != "listed":
                return name
            return areas.region_around(message, str(name)) or name
        except Exception:  # the list can't be read: the name stays as it is
            return name

    details, asked = dict(out["details"]), regional(out["asked_city"])
    if details.get("city"):
        details["city"] = regional(details["city"])
    cities = [str(regional(c)) for c in out.get("cities", [])]
    place = details.get("city") or asked
    relation = out.get("place_relation") or "none"

    if place and relation == "near":
        near = f"{out.get('locality') or 'the area'} near {place}"
        details = {**details, "city": near} if details.get("city") else details
        asked = near if asked else asked
        cities = [near if c.lower() == str(place).lower() else c for c in cities]
        unverified.append(near)
    elif place and relation != "inside" and not (relation == "region" and is_region(str(place))):
        unverified.append(str(place))
        not_stored.add(str(place).lower())
    for name in [details.get("city"), asked, *cities]:
        if name and is_region(str(name)):
            not_stored.add(str(name).lower())
    for value in (details.get("city"), asked):  # a place never stored is still checked
        if value and str(value).lower() in not_stored and value not in cities:
            cities.append(str(value))
    if details.get("city") and str(details["city"]).lower() in not_stored:
        details.pop("city")
    if asked and str(asked).lower() in not_stored:
        asked = None
    return {**out, "details": details, "asked_city": asked, "cities": cities}, unverified, not_stored


def check_named_cities(names: list[str], areas_fn: Callable[[], ServiceAreas],
                       unverified: list[str] = ()) -> list[dict]:
    """The service-area result (the check_service_area tool's data) for every place named, once each, at most
    MAX_CITY_CHECKS. T13e: a place in `unverified` (only near a city, or unclear) is unverified without a lookup. Empty
    when the list can't be read; the model can still call the tool then."""
    checks, seen = [], set()
    unverified_keys = {name.lower() for name in unverified}
    for name in names:
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        if name.lower() in unverified_keys:
            checks.append(unverified_area_data(name))
        else:
            try:
                checks.append(service_area_data(areas_fn(), name))
            except Exception as exc:  # DB down or not ingested
                logger.warning("service-area check before the model unavailable: %s", type(exc).__name__)
                break
        if len(checks) == MAX_CITY_CHECKS:
            break
    return checks


def area_checks_text(checks: list[dict]) -> str:
    """The checks for the agent's message: one JSON line per city, the same data the tool returns."""
    return "\n".join(json.dumps(c) for c in checks)


def package_guidance(case: str | None) -> str:
    """The package rule for the agent's message on a bill turn (T13b: prompts/package_guidance_prompt.md with the
    property type "home", "business" or "unknown"), or "" when the turn is not about packages (`package_case`)."""
    return load_prompt("package_guidance_prompt").replace("{property}", case) if case else ""


def tool_outcome_reply(scratch: list[BaseMessage], written: list[dict] = (), exploring: bool = False) -> str | None:
    """What this turn wrote, in fixed words: the tools' results in `scratch` plus `written` (T13d: the code's own save in
    finalize, as artifacts of the same shape). For when the model fails after them (T12b: in the slow-Google drill Groq
    ran out of tokens on the call after a tool round, and "try again" after a saved lead is false) and for a lead the
    code saved after the model answered. None when nothing was written. T13e: for a customer only exploring, the saved
    line says the team can help whenever they are ready and asks no callback time."""
    lines = []
    artifacts = [m.artifact for m in scratch if isinstance(m, ToolMessage) and isinstance(m.artifact, dict)]
    for artifact in artifacts + list(written):
        if artifact.get("kind") == "lead" and artifact.get("created"):
            lines.append(welcome.SAVED_EXPLORING_TEXT if exploring else welcome.SAVED_FALLBACK_TEXT)
        elif artifact.get("kind") == "callback":
            text = welcome.CALLBACK_UPDATED_TEXT if artifact.get("updated") else welcome.CALLBACK_LOGGED_TEXT
            lines.append(text.format(time=artifact["time"]))
        elif artifact.get("kind") == "support":
            lines.append(welcome.SUPPORT_SAVED_FALLBACK_TEXT.format(id=artifact["id"]))
    if any("callback request" in line for line in lines):  # the callback is logged: no need to ask for a time
        lines = [line.replace(" What time would suit you for a callback?", "") for line in lines]
    return " ".join(lines) or None


def create_lead_refused(scratch: list[BaseMessage]) -> bool:
    """True when create_lead ran this turn and saved nothing (details missing or invalid). The rest of the turn then
    goes without it (T13c: chat 1 called it twice in one turn, "missing: city" both times, and the turn took 13.4 s)."""
    for message in scratch:
        if isinstance(message, ToolMessage) and message.name == "create_lead":
            try:
                if json.loads(str(message.content)).get("saved") is False:
                    return True
            except (ValueError, AttributeError):
                continue
    return False


def _dedupe(chunks: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for chunk in chunks:
        if chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            unique.append(chunk)
    return unique


def tool_area_checks(scratch: list[BaseMessage]) -> list[dict]:
    """The service-area results of this turn's check_service_area calls (the JSON line before the context)."""
    checks = []
    for message in scratch:
        if isinstance(message, ToolMessage) and message.name == "check_service_area":
            try:
                data = json.loads(str(message.content).split("\n\n", 1)[0])
            except ValueError:
                continue
            if data.get("status") in ("listed", "unverified", "region"):
                checks.append({**data, "by": "tool"})
    return checks


def _tool_chunks(scratch: list[BaseMessage], name: str) -> list[dict]:
    """The KB chunks (artifacts) of this turn's results from the tool `name`."""
    return [c for m in scratch if isinstance(m, ToolMessage) and m.name == name for c in (m.artifact or [])]


def build_graph(deps: AgentDeps):
    """Compile the agent graph with the given dependencies."""
    tool_node = ToolNode(deps.tools, messages_key="scratch")
    tool_names = [t.name for t in deps.tools]

    def city_key(name: str) -> str:
        """When two names are the same city: the service-area match (Gurgaon is Gurugram), else the name."""
        try:
            return (deps.areas_fn().lookup(name).matched_city or name).lower()
        except Exception:  # the list can't be read: compare the names
            return name.lower()

    def understand_node(state: AgentState) -> dict:
        message = state["messages"][-1].content
        history = state["messages"][:-1]
        previous = state["messages"][-2] if len(state["messages"]) > 1 else None
        from_text: dict = {}
        try:
            areas = deps.areas_fn()
        except Exception:  # the list can't be read: the places are checked later by the model's tool, if at all
            areas = None
        try:
            out = understand(deps.understand_llm, deps.hints_fn(), message, history)
            degraded = False
        except LLMError as exc:
            # T13d: the KB search always runs on the raw message (Hinglish questions rarely end in "?"), and the details
            # are read in code, so a failed turn keeps the phone, email, city, bill and size it was given. T13e: every
            # reading of the customer's words in this node is the backup for this failed step only
            logger.warning("understand failed, using the raw message: %s", type(exc).__name__)
            from_text = details_from_text(str(message), areas)
            short_answer = short_answer_details(previous, str(message), from_text)  # "3500" after a bill question
            asks_call = asks_to_be_called(str(message))  # a request to be called is ready intent (T13d)
            bare = short_reply(str(message)) is not None  # "ok", "thanks", "no": a reply to us, nothing to search
            out = {"intents": (["reply_to_assistant"] if bare else ["question"])
                   + (["gives_details"] if from_text else []) + (["contact_request"] if asks_call else []),
                   "search_query": None if bare else message,
                   "details": from_text, "asked_city": None, "cities": [], "callback_time": None, "issue": None,
                   "project_details": None, "declines": False,
                   "reply_type": reply_type_from_words(str(message), previous, {**from_text, **short_answer}),
                   "ruled_out": ruled_out_details(str(message)), "locality": None,
                   "place_relation": "inside" if from_text.get("city") else "none",
                   "asks_offerings": asks_what_is_offered(str(message)), "problem_now": "no",
                   "provider": None, "model": None, "latency_ms": None, "tokens": None}
            degraded = True
        # T13e: only a problem the customer's system has now starts the support flow (the understand step's
        # problem_now): a what-if is a question, and an unclear one gets the general support answer and the question
        # whether something is wrong now (Adithya's live test: "agar inverter kharab ho jaye…" opened a support case)
        problem_now = out.get("problem_now") or "no"
        if problem_now == "yes" and not (out.get("issue") or state.get("support", {}).get("issue")):
            problem_now = "unsure"  # "yes" needs the problem itself: the AI's issue, from this message or an earlier one
        if problem_now != "yes" and "existing_customer_issue" in out["intents"]:
            kept = [i for i in out["intents"] if i != "existing_customer_issue"]
            out = {**out, "intents": kept + ([] if "question" in kept else ["question"]), "issue": None}
        elif problem_now == "yes" and "existing_customer_issue" not in out["intents"]:
            out = {**out, "intents": [*out["intents"], "existing_customer_issue"]}
        # T13e: the place as the understand step split it; code confirms only a place inside a listed city
        out, unverified, not_stored = place_rules(out, areas, str(message))
        short = short_answer_details(previous, str(message), out["details"]) if degraded else {}  # backup (T12a)
        data, errors = LeadData.from_raw({**out["details"], **short})
        clean = data.model_dump(exclude_none=True)
        # "4pm" is a time to call, never an installation timeline (T12a replay: the callback time was lost)
        callback_time = merge_callback_time(state.get("callback_time"), out["callback_time"])
        if clean.get("timeline") and is_clock_time(clean["timeline"]):
            callback_time = merge_callback_time(callback_time, clean.pop("timeline"))
            clean.pop("timeline_bucket", None)
        if degraded and not out["callback_time"] and is_clock_time(str(message)):  # backup: a message only "4pm"
            callback_time = merge_callback_time(callback_time, str(message).strip().rstrip(".!"))
        intents_list, search_query, query_note = list(out["intents"]), out["search_query"], None
        # A question always searches: with no query from the model, the raw message is the query (q13 in T12a)
        skip_raw = {"off_topic", "reply_to_assistant", "greeting", "thanks_or_bye"} | NO_SEARCH_TRIGGERS
        if "question" in intents_list and not search_query and not set(intents_list) & skip_raw:
            search_query = str(message)
            query_note = "no query from the model: the message is the query"
        intents = set(intents_list)
        support = dict(state.get("support", {}))
        for key in ("issue", "project_details"):
            if out[key]:
                support[key] = out[key]
        support_case = bool(state.get("support_case")) or "existing_customer_issue" in intents
        if "buying_interest" in intents and "existing_customer_issue" not in intents:
            support_case = False  # the customer moved on to buying: sales rules apply again
        # T13a: a city the customer only asked about counts as theirs; two or more in question: none is guessed. T13e: a
        # region or an unclear place never counts as their city
        before = state.get("lead", {})
        asked = list(state.get("asked_cities", []))
        stated = clean.get("city") or confirmed_city(asked, previous, out)
        named_now = [str(n) for n in [*out.get("cities", []), out["asked_city"]]
                     if n and str(n).lower() not in not_stored]
        asks = [] if intents <= NO_CITY_CHECK_INTENTS or support_case else [
            n for n in named_now if not stated or n.lower() != str(stated).lower()]
        city, city_source, asked = resolve_city(
            current=before.get("city"), source=state.get("city_source"), candidates=asked, stated=stated,
            asked=asks, saved=bool(state.get("lead_id")), key=city_key)
        if city and city != clean.get("city") and city != before.get("city"):
            city = LeadData.from_raw({"city": city})[0].city  # an asked city is checked like a stated one
            if city:
                clean["city"] = city
        # T12b: every place named is checked in code; its step and the KB Section 10 summary (the source) go along
        named = [str(n) for n in [*out.get("cities", []), out["asked_city"], out["details"].get("city")] if n]
        checks = [] if intents <= NO_CITY_CHECK_INTENTS else check_named_cities(named, deps.areas_fn, unverified)
        area_steps = [{"node": "area_check", "detail": f"{c['city']}: {c['status']}", "data": {"by": "code"}}
                      for c in checks]
        summary = deps.hints_fn().summaries.get(SERVICE_AREA_SECTION) if checks else None
        # T13b: a home or a business with no customer type yet. T13e: the understand step decides (its place rules);
        # the place words of the message are read only when that step failed
        words = place_type(str(message), previous) if degraded else {}
        kind = {} if clean.get("customer_type") or before.get("customer_type") else words
        clean.update(kind)
        # T13c: buying interest or a callback request counts as "interested" while the intent is unknown
        intent = chat_intent({**before, **clean}, intents, callback_time)
        clean.update(intent)
        lead = {**before, **clean}
        if not city and asked:  # two or more cities in question: none is guessed (T13a)
            lead.pop("city", None)
        new_details = changed_fields(before, lead)
        # T13a: a monthly bill given this turn brings KB Section 9 (which reference package fits which need), once the
        # property type is known (T13b; package_case)
        packages = package_case({"lead": lead, "new_details": new_details, "support_case": support_case,
                                 "contact_request": "contact_request" in intents})
        guidance = deps.hints_fn().guidance if packages in ("home", "business") else []
        # T13d, T13e: the KB Section 8 summary joins when a size is new this message, or when the understand step's own
        # query is about the catalog (its title or an offering named), so a 5 kW or "what do you offer" answer can name
        # every package; the message's words are read only when that step failed
        catalog_chunk = deps.hints_fn().summaries.get(CATALOG_SECTION)
        asks_offerings = bool(out.get("asks_offerings")) and not support_case  # T13e: the understand step's flag
        if degraded:
            wants_catalog = bool(SIZE_RE.search(str(message)) or asks_offerings)
        else:
            wants_catalog = ("system_size_kw" in new_details or asks_offerings
                             or names_catalog(search_query, catalog_chunk))
        catalog = [catalog_chunk] if catalog_chunk and wants_catalog and not support_case else []
        # T13d, T13e: the questions and offers the customer skipped or declined in a row, from the understand step's
        # reply_type and our own last reply (a clear no to an offer or to sharing details parks asks at once)
        reply_type = out["reply_type"]
        new_ruled = sorted(set(out["ruled_out"]) - set(state.get("ruled_out", [])))
        new_callback = bool(callback_time and callback_time != state.get("callback_time"))
        # naming a region (our own area check says so) answers a question about their city
        reset = (reply_type in ("answers", "accepts", "goes_ahead") or new_callback
                 or bool(intents & {"contact_request", "asks_what_is_needed"})
                 or any(c.get("status") == "region" for c in checks))
        parks = (reply_type == "declines" and (made_offer(previous) or asked_contact(previous))
                 and not new_details and not new_ruled)
        skipped = skipped_asks(state, previous_asked(state, previous), reply_type, reset, parks)
        # T13e: the closing text only when nothing else in the message needs an answer ("closes" skips the model, so a
        # question or a detail would be dropped); a thanks that also gives a detail or asks something goes to the model
        says_bye = reply_type == "closes" or intents == {"thanks_or_bye"}
        closes = (says_bye and not search_query and not new_details and not new_callback
                  and intents <= {"thanks_or_bye", "reply_to_assistant", "greeting"})
        if closes:
            intents_list = ["thanks_or_bye"]
        elif "thanks_or_bye" in intents_list:
            intents_list = [i for i in intents_list if i != "thanks_or_bye"] or (
                ["gives_details"] if new_details else ["question"] if search_query else ["reply_to_assistant"])
        step = {
            "node": "understand",
            "detail": ("failed, raw message used" if degraded else "intents: " + ", ".join(intents_list)),
            "data": {
                "intents": intents_list,
                "search_query": search_query,
                "query_note": query_note,
                "reply_type": reply_type,
                "closes": closes,
                "problem_now": problem_now,
                "asks_offerings": asks_offerings,
                "place_relation": out.get("place_relation"),
                "locality": out.get("locality"),
                "details": sorted(clean),
                "invalid": sorted(errors),
                "city_source": city_source,
                "short_answer": sorted(short),
                "from_words": sorted(kind),
                "intent_from_chat": bool(intent),
                "from_text": sorted(from_text),
                "catalog": bool(catalog),
                "skipped_asks": skipped,
                "provider": out["provider"],
                "model": out["model"],
                "latency_ms": out["latency_ms"],
                "tokens": out["tokens"],
            },
        }
        return {
            "intents": intents_list,
            "search_query": search_query,
            "lead": lead,
            "new_details": new_details,
            "declines": bool(out["declines"]) or reply_type == "declines",
            "ruled_out": sorted(set(state.get("ruled_out", [])) | set(out["ruled_out"])),
            "validation_errors": errors,
            "asked_cities": asked,
            "city_source": city_source,
            # "4pm" then "day after tomorrow" -> "day after tomorrow, 4pm" (T12a)
            "callback_time": callback_time,
            "support": support,
            "support_case": support_case,
            "contact_request": "contact_request" in intents,
            "degraded": degraded,
            "area_checks": checks,
            # T13e: for "what do you offer?" the catalog comes first, so the answer starts from the offerings
            "chunks": (catalog + ([summary] if summary else []) + guidance if asks_offerings
                       else ([summary] if summary else []) + guidance + catalog),
            "asks_offerings": asks_offerings,
            "problem_unclear": problem_now == "unsure" and not support_case,
            "skipped_asks": skipped,
            "ever_declined": bool(state.get("ever_declined")) or bool(out["declines"]) or reply_type == "declines",
            "call_requested": bool(state.get("call_requested")) or "contact_request" in intents,
            "steps": state["steps"] + [step] + area_steps,
        }

    def route_after_understand(state: AgentState) -> str:
        intents = set(state["intents"])
        if intents == {"greeting"}:
            return "welcome"
        if intents == {"thanks_or_bye"}:
            return "closing"
        if intents == {"off_topic"} and not state["search_query"]:
            return "redirect"
        if intents <= NO_SEARCH_INTENTS and intents & NO_SEARCH_TRIGGERS:  # details or contact request only
            return "agent"
        return "retrieve" if state["search_query"] else "agent"

    def retrieve_node(state: AgentState) -> dict:
        start = time.perf_counter()
        result = deps.search_fn(state["search_query"])  # DB errors propagate: the API answers 503
        found = [] if result.no_context else [asdict(c) for c in result.chunks]
        chunks = found
        if chunks and deps.small_to_big:
            chunks = add_table_summaries(chunks, deps.hints_fn().summaries)
        with_summaries = len(chunks)
        if chunks and not state.get("support_case"):  # T13c: part of the package list found: all of it joins
            message = str(state["messages"][-1].content)
            chunks = add_all_packages(chunks, deps.hints_fn().guidance, state.get("lead", {}), message)
        step = retrieve_step(result)
        step["data"]["latency_ms"] = round((time.perf_counter() - start) * 1000)
        step["data"]["added_summaries"] = with_summaries - len(found)
        step["data"]["added_package_rows"] = len(chunks) - with_summaries
        return {
            # a city check's KB Section 10 summary stays first, so it is that answer's source (T12b, as T10b)
            "chunks": _dedupe(state.get("chunks", []) + chunks),
            "no_context": result.no_context,
            "search_mode": result.mode,
            "degraded": state["degraded"] or result.degraded,
            "steps": state["steps"] + [step],
        }

    def agent_messages(state: AgentState) -> list[BaseMessage]:
        persisted = state["messages"][-AGENT_HISTORY:]
        current = persisted[-1]
        chunks = [RetrievedChunk(**c) for c in state["chunks"]]
        lead_progress = status_block(state, offers_made(state["messages"])) if deps.ops is not None else ""
        return [
            SystemMessage(load_prompt("system_prompt")),
            *persisted[:-1],
            rag_message(str(current.content), chunks, lead_progress, area_checks_text(state.get("area_checks", [])),
                        package_guidance(package_case(state))),
            *state.get("scratch", []),
        ]

    def agent_node(state: AgentState) -> dict:
        rounds = state.get("tool_rounds", 0)
        llm = deps.agent_llm if rounds < MAX_TOOL_ROUNDS else deps.final_llm
        offer_create = True
        if (llm is deps.agent_llm and deps.agent_llm_without_create is not None
                and create_lead_refused(state.get("scratch", []))):  # T13c: create_lead once a turn when it can't save
            llm, offer_create = deps.agent_llm_without_create, False
        start = time.perf_counter()
        messages = agent_messages(state)
        message, last_try = None, False
        try:
            message = llm.invoke(messages)
        except Exception as exc:
            logger.warning("agent call failed: %s", type(exc).__name__)
            # T13d: when no tool wrote anything, one last try (a paused model whose pause ends soon, or the main model
            # with a longer deadline) inside the turn's 40 s, before finalize falls back to the safety net
            if getattr(llm, "last_try", None) is not None and tool_outcome_reply(state.get("scratch", [])) is None:
                try:
                    message, last_try = llm.last_try.invoke(messages), True
                except Exception as again:
                    logger.warning("last try failed: %s", type(again).__name__)
        if message is None:  # finalize says what was saved, or answers from the KB chunks (T13d)
            step = {"node": "agent", "detail": "LLM failed", "data": {"provider": None, "latency_ms": None}}
            return {"model_failed": True, "degraded": True, "steps": state["steps"] + [step]}
        latency_ms = round((time.perf_counter() - start) * 1000)
        provider = provider_of(message)
        calls = [c["name"] for c in getattr(message, "tool_calls", [])]
        detail = ("last try: " if last_try else "") + f"{provider}, {latency_ms} ms" + (
            f", calls {', '.join(calls)}" if calls else "")
        step = {"node": "agent", "detail": detail,
                "data": {"provider": provider, "model": message.response_metadata.get("model_name"),
                         "latency_ms": latency_ms, "tokens": tokens_of(message), "tool_calls": calls}}
        if not offer_create:
            step["data"]["create_lead_offered"] = False
        if last_try:
            step["data"]["last_try"] = True
        return {
            "scratch": [message],
            "tool_rounds": rounds + (1 if calls else 0),
            "steps": state["steps"] + [step],
        }

    def route_after_agent(state: AgentState) -> str:
        if state.get("reply") or state.get("model_failed"):
            return "finalize"
        last = state["scratch"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "finalize"

    def tools_node(state: AgentState, config) -> dict:
        start = time.perf_counter()
        out = tool_node.invoke(state, config)
        results = out["scratch"]
        update = {"scratch": results}
        for message in results:  # what the lead tools did, from their artifacts (never shown to the model)
            artifact = message.artifact if isinstance(message.artifact, dict) else {}
            if artifact.get("kind") == "lead":
                update["lead_id"] = artifact["lead_id"]
                update["lead_saved"] = artifact["saved_fields"]
                if artifact.get("reason_pending"):
                    update["reason_pending"] = artifact["lead_id"]
                if artifact.get("created"):
                    update["lead_created_now"] = True
            elif artifact.get("kind") == "callback":
                update["callback_status"] = artifact["status"]
                update["callback_logged"] = artifact["time"]
                update["callback_now"] = "updated" if artifact.get("updated") else "requested"
            elif artifact.get("kind") == "support":
                update["support_request_id"] = artifact["id"]
        steps = [tool_step(m) for m in results]
        if steps:
            steps[-1]["data"]["latency_ms"] = round((time.perf_counter() - start) * 1000)
        update["steps"] = state["steps"] + steps
        return update

    def fixed_reply(text: str, detail: str):
        def node(state: AgentState) -> dict:
            step = {"node": "reply", "detail": detail, "data": {"llm": False}}
            return {"reply": text, "steps": state["steps"] + [step]}
        return node

    def model_reply(state: dict, chunks: list[RetrievedChunk], steps: list[dict]) -> str:
        """The agent's last message as the reply, through the safety nets (T12a-T13d). `state` already holds this turn's
        code-side save. Each guard that fires adds a step, so a slipping prompt shows."""
        reply = strip_tool_citations(normalize_citations(reply_text(state["scratch"][-1])), tool_names)
        if not reply:
            return ""
        # KB tags only on facts from this turn's sources, and no "saved" claim while no lead exists (T12a)
        reply, removed = strip_unsourced_citations(reply, chunks)
        if removed:
            steps.append({"node": "guard", "detail": f"removed {removed} KB tag(s) not backed by this "
                                                     "turn's sources", "data": {"removed": removed}})
        # T13c: no policy, either way, on what the KB doesn't cover (discounts, offers, brands)
        reply, uncovered = remove_uncovered_policies(reply, chunks, welcome.UNCOVERED_TEXT)
        if uncovered:
            steps.append({"node": "guard", "detail": "replaced a policy the KB doesn't cover",
                          "data": {"removed": uncovered}})
        # T13d: nothing is free unless the KB says so (Groq wrote "a free consultation" twice on 13 Oct)
        reply, free = remove_free_claims(reply, chunks, welcome.UNCOVERED_TEXT)
        if free:
            steps.append({"node": "guard", "detail": "removed a claim that a service is free",
                          "data": {"removed": free}})
        # T12b: no number, duration, amount or percentage that nothing in the conversation gave (KB Section 16)
        scratch = state.get("scratch", [])
        allowed = ([c.content for c in chunks]
                   + [str(m.content) for m in scratch if isinstance(m, ToolMessage)]
                   + [str(m.content) for m in state["messages"]]  # the customer's and earlier replies
                   + list(state.get("validation_errors", {}).values()))
        reply, numbers, cut = remove_unsupported_numbers(reply, allowed, welcome.NUMBERS_CONFIRM_TEXT)
        if numbers:  # the removed sentences go into the step (not the log), so each one can be checked
            steps.append({"node": "guard", "detail": f"removed {len(numbers)} number(s) no source gave",
                          "data": {"removed": cut, "numbers": numbers}})
        # T13c: with no size settled, every reference package of a kind or none (the packages the customer
        # named in this message always stay)
        reply, partial = all_or_no_packages(reply, reference_packages(deps.hints_fn().guidance),
                                            state.get("lead", {}), str(state["messages"][-1].content),
                                            welcome.PACKAGES_FIT_TEXT)
        if partial:
            steps.append({"node": "guard", "detail": "removed a partial list of reference packages",
                          "data": {"removed": partial}})
        reply, fixed = remove_false_claims(reply, state) if deps.ops is not None else (reply, False)
        if fixed:
            steps.append({"node": "guard", "detail": "removed a claim that details were saved (no lead "
                                                     "yet)", "data": {}})
        reply, matched = fix_callback_claims(reply, state) if deps.ops is not None else (reply, False)
        if matched:
            steps.append({"node": "guard", "detail": "the callback sentence now says what was logged",
                          "data": {}})
        reply, softened = soften_promises(reply)
        if softened:
            steps.append({"node": "guard", "detail": "a promise that the team will act became 'can'",
                          "data": {}})
        reply, plain = plain_service_area_words(reply)
        if plain:
            steps.append({"node": "guard", "detail": "'reference locations' became 'service areas'",
                          "data": {}})
        # T13e: never "the knowledge base" to the customer, and never a KB rule written for the assistant
        reply, kb_words = plain_kb_words(reply)
        if kb_words:
            steps.append({"node": "guard", "detail": "'the knowledge base' became plain words", "data": {}})
        reply, rules = remove_kb_rules(reply)
        if rules:
            steps.append({"node": "guard", "detail": "removed a KB rule quoted to the customer",
                          "data": {"removed": rules}})
        # T13e: a region is never confirmed: its listed cities, then which city
        reply, region = fix_region_reply(reply, state, chunks)
        if region:
            steps.append({"node": "guard", "detail": "a region is not a city: its cities, then which city",
                          "data": {"changed": region}})
        reply, city_asks = drop_known_city_question(reply, state)
        if city_asks:  # the removed sentences are kept in the step, so each removal can be checked (T13a)
            steps.append({"node": "guard", "detail": "removed a question about a city it already has",
                          "data": {"removed": city_asks}})
        reply, parked = drop_detail_questions(reply, state)
        if parked:  # T13d: the customer skipped two questions about their details in a row
            steps.append({"node": "guard", "detail": "removed a question for details after two skipped ones",
                          "data": {"removed": parked}})
        # T13e: no repeated question, each offer once per chat, none while parked, no callback ask for an explorer,
        # no contact ask on an info-only turn, one detail per question
        reply, asked = ask_rules(reply, state)
        if asked:
            steps.append({"node": "guard", "detail": "question rules: " + "; ".join(a["why"] for a in asked),
                          "data": {"changed": asked}})
        reply, trimmed = keep_one_question(reply, state)
        if trimmed:
            steps.append({"node": "guard", "detail": "kept one question", "data": {}})
        # T13e: "what do you offer?" (the understand step's flag) always names the KB Section 8 offerings: a reply that
        # names none of them gets the list from the KB, with its tag (the assignment's first example question)
        if state.get("asks_offerings"):
            catalog = next((c for c in chunks if c.section_no == CATALOG_SECTION and c.chunk_type == "table_summary"),
                           None)
            if catalog is not None:
                reply, listed = add_offerings_line(reply, catalog.content, welcome.OFFERINGS_INTRO,
                                                   citation_label(catalog))
                if listed:
                    steps.append({"node": "guard", "detail": "added the KB Section 8 offerings the reply left out",
                                  "data": {}})
        # T13e: naming the package for a stated size says a site assessment confirms the final size (KB Section 9)
        message = str(state["messages"][-1].content)
        if "system_size_kw" in state.get("new_details", []) or SIZE_RE.search(message):
            catalog = next((c for c in chunks if c.section_no == CATALOG_SECTION), None)
            reply, sized = add_size_line(reply, reference_packages(deps.hints_fn().guidance),
                                         welcome.SIZE_ASSESSMENT_TEXT, citation_label(catalog) if catalog else "")
            if sized:
                steps.append({"node": "guard", "detail": "added that a site assessment confirms the final size",
                              "data": {}})
        # T13d: a reply that used this turn's chunks without citing any gets the best-matching chunk's tag, so its
        # source chip shows (Groq often leaves the tag out); never on the fixed lines the guards put in
        fixed_lines = (welcome.UNCOVERED_TEXT, welcome.NUMBERS_CONFIRM_TEXT, welcome.PACKAGES_FIT_TEXT,
                       welcome.EXPLORER_READY_TEXT, welcome.SIZE_ASSESSMENT_TEXT)
        reply, tag = add_missing_tag(reply, chunks, skip=fixed_lines)
        if tag:
            steps.append({"node": "guard", "detail": "added the KB tag the reply left out", "data": {"tag": tag}})
        # T13c: never a blank bubble. The number guard removes a sentence with a count no source gave ("8 steps",
        # "7 kW") without a replacement, so a reply made only of such sentences came out empty
        if not reply.strip():
            logger.warning("reply empty after the guards; the fixed line replaced it")
            reply = welcome.NUMBERS_CONFIRM_TEXT
            steps.append({"node": "guard", "detail": "nothing was left after the guards; the fixed line "
                                                     "replaced it", "data": {}})
        return reply

    def finalize_node(state: AgentState) -> dict:
        steps = list(state["steps"])
        update: dict = {}
        written: list[dict] = []  # what the code wrote this turn, shaped like the tools' artifacts
        if deps.ops is not None:
            # T13d: a lead with nothing required missing is saved here when the model didn't call create_lead (its
            # call failed, or it asked again for a detail it had); the tool stays the normal path
            saved = save_ready_lead(state, deps.ops)
            if saved:
                update.update(saved)
                written.append({"kind": "lead", "created": True})
                detail = "lead saved (create_lead was not called)"
                if saved.get("callback_now"):
                    written.append({"kind": "callback", "time": saved["callback_logged"],
                                    "updated": saved["callback_now"] == "updated"})
                    detail += "; callback request logged"
                steps.append({"node": "save", "detail": detail, "data": {"lead_id": saved["lead_id"]}})
        view = {**state, **update}
        scratch = state.get("scratch", [])
        # The service-area summary first: a section's first chunk is its source, and for a city answer the KB
        # Section 10 summary supports it, not whichever row the search found (T10b)
        found = _dedupe(_tool_chunks(scratch, "check_service_area") + state["chunks"]
                        + _tool_chunks(scratch, "search_knowledge_base"))
        chunks = [RetrievedChunk(**c) for c in found]
        degraded = state["degraded"]
        outcome = tool_outcome_reply(scratch, written, exploring=is_explorer(view))

        def safety_net() -> tuple[str, str]:
            """T13d: no model answered: the KB sentences, the city checks, what was saved and one question, instead of
            "I'm having trouble right now". A city the model checked with the tool counts like one the code checked."""
            checks = list(state.get("area_checks", [])) + [c for c in tool_area_checks(scratch)
                                                            if c not in state.get("area_checks", [])]
            text, extra, sections = fallback_reply({**view, "area_checks": checks}, found, outcome)
            update.update(extra)
            detail = ("no model answered: reply from KB Section" + ("s " if len(sections) > 1 else " ")
                      + ", ".join(map(str, sections))) if sections else "no model answered: reply in fixed words"
            steps.append({"node": "fallback", "detail": detail, "data": {"llm": False, "sections": sections}})
            return text, "knowledge_base" if "[KB Section" in text else ("tools" if outcome else "fixed")

        if state.get("model_failed"):  # every answer model failed, and the last try too
            reply, answered_by = safety_net()
        elif state.get("reply"):  # a fixed reply: welcome, closing, redirect
            reply, answered_by = (outcome, "tools") if written else (state["reply"], "fixed")
        else:
            reply = model_reply(view, chunks, steps)
            answered_by = "model"
            if not reply:  # an empty answer is no answer
                degraded = True
                reply, answered_by = safety_net()
            elif update.get("lead_created_now") and outcome:  # saved in code after the model answered
                reply = add_saved_line(reply, outcome)
        reply = tidy_citations(reply, chunks)
        sources = pick_sources(reply, chunks) if chunks and answered_by != "fixed" else []
        if deps.ops is not None:
            changes = pending_changes(view.get("lead", {}), view.get("lead_saved", {}))
            # new details the update_lead tool did not save this turn; T13d: or a call asked for (with no time) after the
            # save, so the action says to call them
            asked_now = (state.get("contact_request") and not view.get("callback_time")
                         and not update.get("lead_created_now"))
            if view.get("lead_id") and (changes or asked_now):
                start = time.perf_counter()
                result = service.update_lead(deps.ops.engine, view["lead_id"], changes,
                                             llm=deps.ops.reason_llm, reason_now=deps.ops.reason_now,
                                             call_requested=bool(view.get("call_requested")))
                deps.ops.calls.append("sync")
                if result.lead.reason_source == "template" and not deps.ops.reason_now:
                    update["reason_pending"] = view["lead_id"]
                update["lead_saved"] = dict(view["lead"])
                steps.append({"node": "sync", "detail": ("saved: " + ", ".join(sorted(changes))) if changes
                              else "the action now says to call them",
                              "data": {"latency_ms": round((time.perf_counter() - start) * 1000)}})
            update["lead_summary"] = lead_summary({**view, **update}, deps.ops.engine)
        return {"reply": reply, "sources": sources, "degraded": degraded, "steps": steps, "answered_by": answered_by,
                "messages": [AIMessage(reply)], **update}

    graph = StateGraph(AgentState)
    graph.add_node("understand", understand_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("welcome", fixed_reply(welcome.WELCOME_TEXT, "welcome text"))
    graph.add_node("closing", fixed_reply(welcome.CLOSING_TEXT, "closing text"))
    graph.add_node("redirect", fixed_reply(welcome.REDIRECT_TEXT, "off-topic redirect"))
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "understand")
    graph.add_conditional_edges("understand", route_after_understand,
                                ["welcome", "closing", "redirect", "retrieve", "agent"])
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "finalize"])
    graph.add_edge("tools", "agent")
    for name in ("welcome", "closing", "redirect"):
        graph.add_edge(name, "finalize")
    graph.add_edge("finalize", END)
    compiled = graph.compile(checkpointer=deps.checkpointer)
    compiled.lead_ops = deps.ops  # the API uses it to write a pending lead reason after replying
    return compiled


def build_agent(
    engine, checkpointer: BaseCheckpointSaver | None, *, main_model: str | None = None, leads_engine=None
):
    """The agent with the real models, KB search and service areas on `engine`.

    leads_engine: where leads, callbacks and support requests are written (default: `engine`; the live
    tests pass the Neon test branch). main_model overrides LLM_MODEL_MAIN (the live fallback test uses a
    model name that does not exist).
    """
    settings = get_settings()
    main_model = main_model or settings.llm_model_main

    def search_fn(query: str) -> SearchResult:
        return search(query, k=CONTEXT_CHUNKS, engine=engine, embedder=query_embedder())

    def hints_fn() -> KbHints:
        return load_kb_hints(engine)

    # The lead reason (an LLM call of about 3 s) is written after the reply, not inside the turn (T8b)
    ops = LeadOps(engine=leads_engine or engine, areas_fn=get_service_areas, reason_now=False)
    tools = make_tools(search_fn, get_service_areas, hints_fn, ops)
    deps = AgentDeps(
        understand_llm=understand_llm(Understanding),  # main, then fast: 5 s each (T13e)
        agent_llm=chat_llm(main_model, tools=tools),
        agent_llm_without_create=chat_llm(main_model, tools=[t for t in tools if t.name != "create_lead"]),
        final_llm=chat_llm(main_model),
        search_fn=search_fn,
        areas_fn=get_service_areas,
        hints_fn=hints_fn,
        tools=tools,
        checkpointer=checkpointer,
        ops=ops,
    )
    return build_graph(deps)


@lru_cache
def get_agent():
    """The app's agent with Postgres memory (FastAPI dependency; tests override it). None without a DB."""
    from app.agent.memory import get_checkpointer

    engine = get_engine()
    if engine is None:
        return None
    return build_agent(engine, get_checkpointer())

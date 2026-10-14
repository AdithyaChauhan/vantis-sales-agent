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
    drop_known_city_question, fix_callback_claims, is_clock_time, keep_one_question, lead_summary, merge_callback_time,
    own_details_question, package_case, passing_place_kinds, pending_changes, place_type, remove_false_claims,
    resolve_city, ruled_out_details, save_ready_lead, short_answer_details, skipped_asks, status_block,
)
from app.agent.packages import SIZE_RE, add_all_packages, all_or_no_packages, reference_packages
from app.agent.state import AgentState
from app.agent.text_details import (
    asks_to_be_called, asks_what_is_offered, details_from_text, is_go_ahead, is_what_if, reports_problem_now,
)
from app.agent.tools import make_tools, service_area_data, tool_step
from app.agent.understand import KbHints, Understanding, load_kb_hints, understand
from app.core.config import get_settings
from app.core.db import get_engine
from app.core.llm import LLMError, chat_llm, provider_of, query_embedder, reply_text, tokens_of, understand_llm
from app.leads import service
from app.core.prompts import load_prompt
from app.leads.schemas import CUSTOMER_TYPE_OF, LeadData
from app.rag.answer import (
    add_missing_tag, normalize_citations, pick_sources, plain_service_area_words, rag_message, remove_free_claims,
    remove_uncovered_policies, retrieve_step, soften_promises, strip_tool_citations, strip_unsourced_citations,
    tidy_citations,
)
from app.rag.chunker import CATALOG_SECTION, SERVICE_AREA_SECTION, SUMMARY_SECTIONS
from app.rag.numbers import remove_unsupported_numbers
from app.rag.retriever import RetrievedChunk, SearchResult, search
from app.services.service_areas import ServiceAreas, get_service_areas, lookalike_place

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


def add_catalog_for_size(
    chunks: list[dict], message: str, new_details: list[str], summaries: dict[int, dict]
) -> list[dict]:
    """When the customer mentions a system size this turn ("5kW", "3 kW"), add the KB Section 8 summary chunk (every
    package with its kW, read from the KB) if it is not in the context yet, so a 5 kW question can name the 5 kW
    package (T12b; T5 answered "How much does a 5kW system cost?" without naming HomePro 5)."""
    summary = summaries.get(CATALOG_SECTION)
    if not summary or not (SIZE_RE.search(message) or "system_size_kw" in new_details):
        return chunks
    if any(c["chunk_id"] == summary["chunk_id"] for c in chunks):
        return chunks
    return chunks + [summary]


def check_named_cities(names: list[str], areas_fn: Callable[[], ServiceAreas]) -> list[dict]:
    """The service-area result (the check_service_area tool's data) for every city named, once each, at most
    MAX_CITY_CHECKS. Empty when the list can't be read; the model can still call the tool then."""
    checks, seen = [], set()
    for name in names:
        if name.lower() in seen:
            continue
        seen.add(name.lower())
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


def tool_outcome_reply(scratch: list[BaseMessage], written: list[dict] = ()) -> str | None:
    """What this turn wrote, in fixed words: the tools' results in `scratch` plus `written` (T13d: the code's own save in
    finalize, as artifacts of the same shape). For when the model fails after them (T12b: in the slow-Google drill Groq
    ran out of tokens on the call after a tool round, and "try again" after a saved lead is false) and for a lead the
    code saved after the model answered. None when nothing was written."""
    lines = []
    artifacts = [m.artifact for m in scratch if isinstance(m, ToolMessage) and isinstance(m.artifact, dict)]
    for artifact in artifacts + list(written):
        if artifact.get("kind") == "lead" and artifact.get("created"):
            lines.append(welcome.SAVED_FALLBACK_TEXT)
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
            if data.get("status") in ("listed", "unverified"):
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
            out = understand(deps.understand_llm, deps.hints_fn(), message, history)
            degraded = False
        except LLMError as exc:
            # T13d: the KB search always runs on the raw message (Hinglish questions rarely end in "?": "subsidy milti
            # hai kya sarkar se"), and the details are read in code, so a failed turn keeps the phone, email, city, bill
            # and size it was given (and the name, from clear patterns only)
            logger.warning("understand failed, using the raw message: %s", type(exc).__name__)
            try:
                areas = deps.areas_fn()
            except Exception:  # the list can't be read: listed cities aren't found in the words
                areas = None
            from_text = details_from_text(str(message), areas)
            asks_call = asks_to_be_called(str(message))  # a request to be called is ready intent (T13d)
            out = {"intents": ["question"] + (["gives_details"] if from_text else [])
                   + (["contact_request"] if asks_call else []), "search_query": message,
                   "details": from_text, "asked_city": None, "cities": [], "callback_time": None, "issue": None,
                   "project_details": None, "declines": False, "provider": None, "model": None, "latency_ms": None,
                   "tokens": None}
            degraded = True
        # T13d: a what-if ("agar inverter kharab ho jaye to kya karna padega") is a question, not a fault: only a problem
        # the customer has now opens a support request
        what_if = ("existing_customer_issue" in out["intents"] and is_what_if(str(message))
                   and not reports_problem_now(str(message)))
        if what_if:
            kept = [i for i in out["intents"] if i != "existing_customer_issue"]
            out = {**out, "intents": kept + ([] if "question" in kept else ["question"]), "issue": None,
                   "search_query": out["search_query"] or str(message)}
        # T13d: a listed city used only inside another place's name or as a landmark ("Greater Noida", "Lucknow ke paas
        # ek gaon") is that other place, never the listed city, whatever the extractor wrote
        def place(name):
            return (lookalike_place(str(message), str(name)) or name) if name else name

        if out["details"].get("city") or out["asked_city"] or out.get("cities"):
            out = {**out, "details": {**out["details"], **({"city": place(out["details"]["city"])}
                                                          if out["details"].get("city") else {})},
                   "asked_city": place(out["asked_city"]), "cities": [place(n) for n in out.get("cities", [])]}
        short = short_answer_details(previous, str(message), out["details"])  # a bare "3500" or "yes" (T12a)
        data, errors = LeadData.from_raw({**out["details"], **short})
        clean = data.model_dump(exclude_none=True)
        # "4pm" is a time to call, never an installation timeline (T12a replay: the callback time was lost)
        callback_time = merge_callback_time(state.get("callback_time"), out["callback_time"])
        if clean.get("timeline") and is_clock_time(clean["timeline"]):
            callback_time = merge_callback_time(callback_time, clean.pop("timeline"))
            clean.pop("timeline_bucket", None)
        if not out["callback_time"] and is_clock_time(str(message)):  # a message that is only "4pm"
            callback_time = merge_callback_time(callback_time, str(message).strip().rstrip(".!"))
        intents_list, search_query, query_note = list(out["intents"]), out["search_query"], None
        # T12b: "what else do you need?" is about their own details (no KB search); "what do I need for
        # installation?" is a KB question, whatever intent the model gave it
        if "asks_what_is_needed" in intents_list and not own_details_question(str(message)):
            intents_list = [i for i in intents_list if i != "asks_what_is_needed"] or ["question"]
            if "question" not in intents_list:
                intents_list.append("question")
            query_note = "not about their details: searches the KB"
        # A question always searches: with no query from the model, the raw message is the query (q13 in T12a)
        skip_raw = {"off_topic", "reply_to_assistant", "greeting", "thanks_or_bye"} | NO_SEARCH_TRIGGERS
        if "question" in intents_list and not search_query and not set(intents_list) & skip_raw:
            search_query = str(message)
            query_note = query_note or "no query from the model: the message is the query"
        out = {**out, "intents": intents_list, "search_query": search_query}
        intents = set(intents_list)
        support = dict(state.get("support", {}))
        for key in ("issue", "project_details"):
            if out[key]:
                support[key] = out[key]
        support_case = bool(state.get("support_case")) or "existing_customer_issue" in intents
        if "buying_interest" in intents and "existing_customer_issue" not in intents:
            support_case = False  # the customer moved on to buying: sales rules apply again
        # T13a: a city the customer only asked about counts as theirs; two or more in question: none is guessed
        before = state.get("lead", {})
        asked = list(state.get("asked_cities", []))
        stated = clean.get("city") or confirmed_city(asked, previous, out)
        named_now = [str(n) for n in [*out.get("cities", []), out["asked_city"]] if n]
        asks = [] if intents <= NO_CITY_CHECK_INTENTS or support_case else [
            n for n in named_now if not stated or n.lower() != str(stated).lower()]
        city, city_source, asked = resolve_city(
            current=before.get("city"), source=state.get("city_source"), candidates=asked, stated=stated,
            asked=asks, saved=bool(state.get("lead_id")), key=city_key)
        if city and city != clean.get("city") and city != before.get("city"):
            city = LeadData.from_raw({"city": city})[0].city  # an asked city is checked like a stated one
            if city:
                clean["city"] = city
        # T12b: every city named is checked in code; its step and the KB Section 10 summary (the source) go along
        named = [str(n) for n in [*out.get("cities", []), out["asked_city"], out["details"].get("city")] if n]
        checks = [] if intents <= NO_CITY_CHECK_INTENTS else check_named_cities(named, deps.areas_fn)
        area_steps = [{"node": "area_check", "detail": f"{c['city']}: {c['status']}", "data": {"by": "code"}}
                      for c in checks]
        summary = deps.hints_fn().summaries.get(SERVICE_AREA_SECTION) if checks else None
        # T13c: the kind of place only with confidence (place_type). A new type from the extractor is dropped when this
        # message names that kind of place only in passing ("I work from home") and no confident words: the bot asks
        words = place_type(str(message), previous)
        new_type, dropped = clean.get("customer_type"), []
        if (new_type and new_type != before.get("customer_type") and not words
                and new_type in passing_place_kinds(str(message))):
            dropped = ["customer_type"] + (["property_type"] if CUSTOMER_TYPE_OF.get(clean.get("property_type"))
                                           == new_type else [])
            for name in dropped:
                clean.pop(name)
        # T13b: a home or a business named, with no customer type from the extractor or an earlier turn (C7 run 1:
        # "it's for my home" stored nothing on the bill turn), like the T12a short answers
        kind = {} if clean.get("customer_type") or before.get("customer_type") else words
        clean.update(kind)
        # T13c: buying interest or a callback request counts as "interested" while the intent is unknown
        intent = chat_intent({**before, **clean}, intents, callback_time)
        clean.update(intent)
        step = {
            "node": "understand",
            "detail": ("failed, raw message used" if degraded else "intents: " + ", ".join(out["intents"])),
            "data": {
                "intents": out["intents"],
                "search_query": out["search_query"],
                "query_note": query_note,
                "details": sorted(clean),
                "invalid": sorted(errors),
                "city_source": city_source,
                "short_answer": sorted(short),
                "from_words": sorted(kind),
                "dropped": dropped,
                "intent_from_chat": bool(intent),
                "from_text": sorted(from_text),
                "what_if": what_if,
                "provider": out["provider"],
                "model": out["model"],
                "latency_ms": out["latency_ms"],
                "tokens": out["tokens"],
            },
        }
        lead = {**before, **clean}
        if not city and asked:  # two or more cities in question: none is guessed (T13a)
            lead.pop("city", None)
        new_details = changed_fields(before, lead)
        # T13a: a monthly bill given this turn brings KB Section 9 (which reference package fits which need), so the
        # packages named have their source even on a turn without a KB search (in C7 and C7b the number guard had to
        # remove "…fits nicely with our 3 kW residential reference package"). T13b: only once the property type is
        # known, since the reply then names every package for a home or a business; with the type still unknown it
        # asks for it first (package_case), and the type given after the bill brings them too
        packages = package_case({"lead": lead, "new_details": new_details, "support_case": support_case,
                                 "contact_request": "contact_request" in intents})
        guidance = deps.hints_fn().guidance if packages in ("home", "business") else []
        # T13d: a size stated this message brings the KB Section 8 summary even without a KB search ("I want a 5 kW
        # system for my house" gives details only, so it never searched and the reply couldn't name HomePro 5); so does
        # a question about what Vantis offers, with or without the understand step (on 13 Oct the raw search for "What
        # solar systems do you provide?" found only Section 1)
        summaries = deps.hints_fn().summaries
        catalog = [] if support_case else add_catalog_for_size([], str(message), new_details, summaries)
        offer = not support_case and not catalog and asks_what_is_offered(str(message))
        if offer and summaries.get(CATALOG_SECTION):
            catalog = [summaries[CATALOG_SECTION]]
        # T13d: questions about their details the customer skipped in a row; a detail, a contact request, "what else do
        # you need" or a clear buying signal lets the assistant ask again
        invited = (bool(intents & {"contact_request", "asks_what_is_needed"}) or is_go_ahead(str(message))
                   or (lead.get("intent_level") == "ready" and before.get("intent_level") != "ready"))
        skipped = skipped_asks(state, previous, new_details, invited)
        step["data"].update({"catalog_for_size": bool(catalog) and not offer, "catalog_for_offer": bool(catalog) and offer,
                             "skipped_asks": skipped})
        return {
            "intents": out["intents"],
            "search_query": out["search_query"],
            "lead": lead,
            "new_details": new_details,
            "declines": bool(out["declines"]),
            "ruled_out": sorted(set(state.get("ruled_out", [])) | set(ruled_out_details(str(message)))),
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
            "chunks": ([summary] if summary else []) + guidance + catalog,
            "skipped_asks": skipped,
            "ever_declined": bool(state.get("ever_declined")) or bool(out["declines"]),
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
        lead_progress = status_block(state) if deps.ops is not None else ""
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
        reply, city_asks = drop_known_city_question(reply, state)
        if city_asks:  # the removed sentences are kept in the step, so each removal can be checked (T13a)
            steps.append({"node": "guard", "detail": "removed a question about a city it already has",
                          "data": {"removed": city_asks}})
        reply, parked = drop_detail_questions(reply, state)
        if parked:  # T13d: the customer skipped two questions about their details in a row
            steps.append({"node": "guard", "detail": "removed a question for details after two skipped ones",
                          "data": {"removed": parked}})
        reply, trimmed = keep_one_question(reply, state)
        if trimmed:
            steps.append({"node": "guard", "detail": "kept one question", "data": {}})
        # T13d: a reply that used this turn's chunks without citing any gets the best-matching chunk's tag, so its
        # source chip shows (Groq often leaves the tag out); never on the fixed lines the guards put in
        fixed_lines = (welcome.UNCOVERED_TEXT, welcome.NUMBERS_CONFIRM_TEXT, welcome.PACKAGES_FIT_TEXT)
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
        outcome = tool_outcome_reply(scratch, written)

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
        understand_llm=understand_llm(Understanding),  # main, Groq, fast: 4 s each (T12a)
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

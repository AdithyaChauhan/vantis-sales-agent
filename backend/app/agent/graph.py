"""The LangGraph agent for one customer message (PLAN.md part 3.1).

understand ─┬─ only a greeting ─► welcome ─┐
            ├─ only thanks/bye ─► closing ─┤
            ├─ only off-topic, no query ─► redirect ─┤
            ├─ search_query (not details-only) ─► retrieve ─► agent ⇄ tools ─┤
            └─ anything else ─────────────────────────────► agent ⇄ tools ─┴─► finalize

Every external piece (LLMs, search, service areas, KB hints, leads database, checkpointer) comes in
through AgentDeps, so tests run the real graph with fakes. The lead tools (T8b) read the draft from the
state; finalize saves new details of an existing lead once (if the update_lead tool did not already).
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
from app.agent.leads import (
    LeadOps, changed_fields, confirmed_city, fix_callback_claims, is_clock_time, keep_one_question, lead_summary,
    merge_callback_time, own_details_question, pending_changes, remove_false_claims, ruled_out_details,
    short_answer_details, status_block,
)
from app.agent.state import AgentState
from app.agent.tools import make_tools, service_area_data, tool_step
from app.agent.understand import KbHints, Understanding, load_kb_hints, understand
from app.core.config import get_settings
from app.core.db import get_engine
from app.core.llm import LLMError, chat_llm, provider_of, query_embedder, reply_text, tokens_of, understand_llm
from app.leads import service
from app.core.prompts import load_prompt
from app.leads.schemas import LeadData
from app.rag.answer import (
    LLM_DOWN_REPLY, normalize_citations, pick_sources, plain_service_area_words, rag_message, retrieve_step,
    soften_promises, strip_tool_citations, strip_unsourced_citations, tidy_citations,
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
MAX_ASKED_CITIES = 3
# T12b: every city named in the message is checked in code before the model answers (it saves one model round per
# city turn, which Groq's 8K tokens a minute could not carry in the slow-Google drill); not for these intents alone
MAX_CITY_CHECKS = 3
NO_CITY_CHECK_INTENTS = {"greeting", "thanks_or_bye", "off_topic"}
# A system size in the message ("5kW", "3 kW"): the catalog summary joins the context, so the reply can name the
# package of that size (T12b; T5 answered "How much does a 5kW system cost?" without naming HomePro 5)
_SIZE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:kw|kwp|kilowatts?)\b", re.IGNORECASE)


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
    """When the customer mentions a system size this turn, add the KB Section 8 summary chunk (every package with
    its kW, read from the KB) if it is not in the context yet, so a 5 kW question can name the 5 kW package (T12b)."""
    summary = summaries.get(CATALOG_SECTION)
    if not summary or not (_SIZE_RE.search(message) or "system_size_kw" in new_details):
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


def tool_outcome_reply(scratch: list[BaseMessage]) -> str | None:
    """What this turn's tools wrote, in fixed words, for when the model fails after them (T12b): in the slow-Google
    drill Groq ran out of tokens on the call after a tool round, and "try again" after a saved lead is false.
    None when no tool wrote anything (the reply is then the usual "try again")."""
    lines = []
    for message in scratch:
        artifact = message.artifact if isinstance(message, ToolMessage) and isinstance(message.artifact, dict) else {}
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


def _dedupe(chunks: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for chunk in chunks:
        if chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            unique.append(chunk)
    return unique


def _tool_chunks(scratch: list[BaseMessage], name: str) -> list[dict]:
    """The KB chunks (artifacts) of this turn's results from the tool `name`."""
    return [c for m in scratch if isinstance(m, ToolMessage) and m.name == name for c in (m.artifact or [])]


def build_graph(deps: AgentDeps):
    """Compile the agent graph with the given dependencies."""
    tool_node = ToolNode(deps.tools, messages_key="scratch")
    tool_names = [t.name for t in deps.tools]

    def understand_node(state: AgentState) -> dict:
        message = state["messages"][-1].content
        history = state["messages"][:-1]
        previous = state["messages"][-2] if len(state["messages"]) > 1 else None
        try:
            out = understand(deps.understand_llm, deps.hints_fn(), message, history)
            degraded = False
        except LLMError as exc:  # carry on with the raw message as the search query
            logger.warning("understand failed, using the raw message: %s", type(exc).__name__)
            out = {"intents": ["question"], "search_query": message, "details": {}, "asked_city": None, "cities": [],
                   "callback_time": None, "issue": None, "project_details": None, "declines": False,
                   "provider": None, "model": None, "latency_ms": None, "tokens": None}
            degraded = True
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
        asked = list(state.get("asked_cities", []))
        city = confirmed_city(asked, previous, out)
        if city:  # the customer confirmed the city the assistant's question named
            clean["city"] = city
        if clean.get("city"):
            asked = []  # a stated or confirmed city replaces the guesses
        elif out["asked_city"] and out["asked_city"] not in asked:
            asked = (asked + [out["asked_city"]])[-MAX_ASKED_CITIES:]
        support = dict(state.get("support", {}))
        for key in ("issue", "project_details"):
            if out[key]:
                support[key] = out[key]
        support_case = bool(state.get("support_case")) or "existing_customer_issue" in intents
        if "buying_interest" in intents and "existing_customer_issue" not in intents:
            support_case = False  # the customer moved on to buying: sales rules apply again
        # T12b: every city named is checked in code; its step and the KB Section 10 summary (the source) go along
        named = [str(n) for n in [*out.get("cities", []), out["asked_city"], out["details"].get("city")] if n]
        checks = [] if intents <= NO_CITY_CHECK_INTENTS else check_named_cities(named, deps.areas_fn)
        area_steps = [{"node": "area_check", "detail": f"{c['city']}: {c['status']}", "data": {"by": "code"}}
                      for c in checks]
        summary = deps.hints_fn().summaries.get(SERVICE_AREA_SECTION) if checks else None
        step = {
            "node": "understand",
            "detail": ("failed, raw message used" if degraded else "intents: " + ", ".join(out["intents"])),
            "data": {
                "intents": out["intents"],
                "search_query": out["search_query"],
                "query_note": query_note,
                "details": sorted(clean),
                "invalid": sorted(errors),
                "city_confirmed": bool(city),
                "short_answer": sorted(short),
                "provider": out["provider"],
                "model": out["model"],
                "latency_ms": out["latency_ms"],
                "tokens": out["tokens"],
            },
        }
        before = state.get("lead", {})
        lead = {**before, **clean}
        return {
            "intents": out["intents"],
            "search_query": out["search_query"],
            "lead": lead,
            "new_details": changed_fields(before, lead),
            "declines": bool(out["declines"]),
            "ruled_out": sorted(set(state.get("ruled_out", [])) | set(ruled_out_details(str(message)))),
            "validation_errors": errors,
            "asked_cities": asked,
            # "4pm" then "day after tomorrow" -> "day after tomorrow, 4pm" (T12a)
            "callback_time": callback_time,
            "support": support,
            "support_case": support_case,
            "contact_request": "contact_request" in intents,
            "degraded": degraded,
            "area_checks": checks,
            "chunks": [summary] if summary else [],
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
        if chunks:
            message = str(state["messages"][-1].content)
            chunks = add_catalog_for_size(chunks, message, state.get("new_details", []), deps.hints_fn().summaries)
        step = retrieve_step(result)
        step["data"]["latency_ms"] = round((time.perf_counter() - start) * 1000)
        step["data"]["added_summaries"] = len(chunks) - len(found)
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
            rag_message(str(current.content), chunks, lead_progress, area_checks_text(state.get("area_checks", []))),
            *state.get("scratch", []),
        ]

    def agent_node(state: AgentState) -> dict:
        rounds = state.get("tool_rounds", 0)
        llm = deps.agent_llm if rounds < MAX_TOOL_ROUNDS else deps.final_llm
        start = time.perf_counter()
        try:
            message = llm.invoke(agent_messages(state))
        except Exception as exc:
            logger.warning("agent call failed: %s", type(exc).__name__)
            done = tool_outcome_reply(state.get("scratch", []))  # never "try again" after a save (T12b)
            detail = "LLM failed; the reply says what the tools did" if done else "LLM failed"
            step = {"node": "agent", "detail": detail, "data": {"provider": None, "latency_ms": None}}
            return {"reply": done or LLM_DOWN_REPLY, "degraded": True, "steps": state["steps"] + [step]}
        latency_ms = round((time.perf_counter() - start) * 1000)
        provider = provider_of(message)
        calls = [c["name"] for c in getattr(message, "tool_calls", [])]
        detail = f"{provider}, {latency_ms} ms" + (f", calls {', '.join(calls)}" if calls else "")
        step = {"node": "agent", "detail": detail,
                "data": {"provider": provider, "model": message.response_metadata.get("model_name"),
                         "latency_ms": latency_ms, "tokens": tokens_of(message), "tool_calls": calls}}
        return {
            "scratch": [message],
            "tool_rounds": rounds + (1 if calls else 0),
            "steps": state["steps"] + [step],
        }

    def route_after_agent(state: AgentState) -> str:
        if state.get("reply"):
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

    def finalize_node(state: AgentState) -> dict:
        reply = state.get("reply") or ""
        sources: list[dict] = []
        degraded = state["degraded"]
        update: dict = {}
        steps = list(state["steps"])
        if not reply:  # the agent's last message is the answer
            reply = strip_tool_citations(normalize_citations(reply_text(state["scratch"][-1])), tool_names)
            if not reply:
                reply, degraded = LLM_DOWN_REPLY, True
            else:
                # The service-area summary first: a section's first chunk is its source, and for a city
                # answer the KB Section 10 summary supports it, not whichever row the search found (T10b)
                scratch = state.get("scratch", [])
                found = (_tool_chunks(scratch, "check_service_area") + state["chunks"]
                         + _tool_chunks(scratch, "search_knowledge_base"))
                chunks = [RetrievedChunk(**c) for c in _dedupe(found)]
                # The safety nets (T12a): KB tags only on facts from this turn's sources, and no "saved"
                # claim while no lead exists. Each one that fires is a step, so a slipping prompt shows.
                reply, removed = strip_unsourced_citations(reply, chunks)
                if removed:
                    steps.append({"node": "guard", "detail": f"removed {removed} KB tag(s) not backed by this "
                                                             "turn's sources", "data": {"removed": removed}})
                # T12b: no number, duration, amount or percentage that nothing in the conversation gave (KB Section 16)
                allowed = ([c.content for c in chunks]
                           + [str(m.content) for m in scratch if isinstance(m, ToolMessage)]
                           + [str(m.content) for m in state["messages"]]  # the customer's and earlier replies
                           + list(state.get("validation_errors", {}).values()))
                reply, numbers = remove_unsupported_numbers(reply, allowed, welcome.NUMBERS_CONFIRM_TEXT)
                if numbers:
                    steps.append({"node": "guard", "detail": f"removed {len(numbers)} number(s) no source gave",
                                  "data": {"removed": len(numbers)}})
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
                reply, trimmed = keep_one_question(reply, state)
                if trimmed:
                    steps.append({"node": "guard", "detail": "kept one question", "data": {}})
                reply = tidy_citations(reply, chunks)
                sources = pick_sources(reply, chunks) if chunks else []
        if deps.ops is not None:
            changes = pending_changes(state.get("lead", {}), state.get("lead_saved", {}))
            if state.get("lead_id") and changes:  # new details the update_lead tool did not save this turn
                start = time.perf_counter()
                result = service.update_lead(deps.ops.engine, state["lead_id"], changes,
                                             llm=deps.ops.reason_llm, reason_now=deps.ops.reason_now)
                deps.ops.calls.append("sync")
                if result.lead.reason_source == "template" and not deps.ops.reason_now:
                    update["reason_pending"] = state["lead_id"]
                update["lead_saved"] = dict(state["lead"])
                steps.append({"node": "sync", "detail": "saved: " + ", ".join(sorted(changes)),
                              "data": {"latency_ms": round((time.perf_counter() - start) * 1000)}})
            update["lead_summary"] = lead_summary({**state, **update}, deps.ops.engine)
        return {"reply": reply, "sources": sources, "degraded": degraded, "steps": steps,
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
        understand_llm=understand_llm(Understanding),  # main, fast, Groq: 5 s each (T10b)
        agent_llm=chat_llm(main_model, tools=tools),
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

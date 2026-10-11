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
from app.agent.leads import LeadOps, confirmed_city, lead_summary, pending_changes, status_block
from app.agent.state import AgentState
from app.agent.tools import KB_TOOLS, make_tools, tool_step
from app.agent.understand import KbHints, Understanding, load_kb_hints, understand
from app.core.config import get_settings
from app.core.db import get_engine
from app.core.llm import LLMError, chat_llm, provider_of, query_embedder, reply_text
from app.leads import service
from app.core.prompts import load_prompt
from app.leads.schemas import LeadData
from app.rag.answer import LLM_DOWN_REPLY, normalize_citations, pick_sources, rag_message, retrieve_step
from app.rag.chunker import SUMMARY_SECTIONS
from app.rag.retriever import RetrievedChunk, SearchResult, search
from app.services.service_areas import ServiceAreas, get_service_areas

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
AGENT_HISTORY = 12  # persisted messages the agent sees, including the current one
CONTEXT_CHUNKS = 4
# No KB search for a message that only gives details or asks to be contacted (a short "yes please" may
# come with it); a short reply alone that accepts an info offer still searches.
NO_SEARCH_INTENTS = {"gives_details", "contact_request", "reply_to_assistant"}
NO_SEARCH_TRIGGERS = {"gives_details", "contact_request"}
MAX_ASKED_CITIES = 3


@dataclass
class AgentDeps:
    """Everything the graph needs from outside."""

    understand_llm: Runnable  # fast model with structured output (Understanding, include_raw)
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


def _dedupe(chunks: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for chunk in chunks:
        if chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            unique.append(chunk)
    return unique


def build_graph(deps: AgentDeps):
    """Compile the agent graph with the given dependencies."""
    tool_node = ToolNode(deps.tools, messages_key="scratch")

    def understand_node(state: AgentState) -> dict:
        message = state["messages"][-1].content
        history = state["messages"][:-1]
        try:
            out = understand(deps.understand_llm, deps.hints_fn(), message, history)
            degraded = False
        except LLMError as exc:  # carry on with the raw message as the search query
            logger.warning("understand failed, using the raw message: %s", type(exc).__name__)
            out = {"intents": ["question"], "search_query": message, "details": {}, "asked_city": None,
                   "callback_time": None, "issue": None, "project_details": None, "declines": False,
                   "provider": None, "latency_ms": None}
            degraded = True
        data, errors = LeadData.from_raw(out["details"])
        clean = data.model_dump(exclude_none=True)
        intents = set(out["intents"])
        asked = list(state.get("asked_cities", []))
        previous = state["messages"][-2] if len(state["messages"]) > 1 else None
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
        step = {
            "node": "understand",
            "detail": ("failed, raw message used" if degraded else "intents: " + ", ".join(out["intents"])),
            "data": {
                "intents": out["intents"],
                "search_query": out["search_query"],
                "details": sorted(clean),
                "invalid": sorted(errors),
                "city_confirmed": bool(city),
                "provider": out["provider"],
                "latency_ms": out["latency_ms"],
            },
        }
        return {
            "intents": out["intents"],
            "search_query": out["search_query"],
            "lead": {**state.get("lead", {}), **clean},
            "validation_errors": errors,
            "asked_cities": asked,
            "callback_time": out["callback_time"] or state.get("callback_time"),
            "support": support,
            "support_case": support_case,
            "contact_request": "contact_request" in intents,
            "degraded": degraded,
            "steps": state["steps"] + [step],
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
        chunks = [] if result.no_context else [asdict(c) for c in result.chunks]
        found = len(chunks)
        if chunks and deps.small_to_big:
            chunks = add_table_summaries(chunks, deps.hints_fn().summaries)
        step = retrieve_step(result)
        step["data"]["latency_ms"] = round((time.perf_counter() - start) * 1000)
        step["data"]["added_summaries"] = len(chunks) - found
        return {
            "chunks": chunks,
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
            rag_message(str(current.content), chunks, lead_progress),
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
            step = {"node": "agent", "detail": "LLM failed", "data": {"provider": None, "latency_ms": None}}
            return {"reply": LLM_DOWN_REPLY, "degraded": True, "steps": state["steps"] + [step]}
        latency_ms = round((time.perf_counter() - start) * 1000)
        provider = provider_of(message)
        calls = [c["name"] for c in getattr(message, "tool_calls", [])]
        detail = f"{provider}, {latency_ms} ms" + (f", calls {', '.join(calls)}" if calls else "")
        step = {"node": "agent", "detail": detail,
                "data": {"provider": provider, "latency_ms": latency_ms, "tool_calls": calls}}
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
            elif artifact.get("kind") == "callback":
                update["callback_status"] = artifact["status"]
                update["callback_logged"] = artifact["time"]
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
            reply = normalize_citations(reply_text(state["scratch"][-1]))
            if not reply:
                reply, degraded = LLM_DOWN_REPLY, True
            else:
                found = state["chunks"] + [
                    c for m in state.get("scratch", [])
                    if isinstance(m, ToolMessage) and m.name in KB_TOOLS for c in (m.artifact or [])
                ]
                chunks = [RetrievedChunk(**c) for c in _dedupe(found)]
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
        understand_llm=chat_llm(settings.llm_model_fast, schema=Understanding),
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

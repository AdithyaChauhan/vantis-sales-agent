"""Tools the agent can call: knowledge search and the service-area check (T8a), and the lead tools (T8b).

Each tool returns (content for the LLM, artifact). The KB tools' artifact holds KB chunk dicts, so the
reply's sources can include what a tool found. The lead tools take no arguments: they read the validated
draft from the graph state (InjectedState), so the model cannot save made-up details, and their results
never contain a score, a status or a reason. The tools return data only; the system prompt says how to
word it.
"""
import json
import logging
from dataclasses import asdict
from typing import Annotated, Callable

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.agent.leads import LeadOps, ordered_missing, pending_changes, support_missing
from app.agent.understand import KbHints
from app.leads import service
from app.rag.answer import format_context
from app.rag.chunker import SERVICE_AREA_SECTION
from app.rag.retriever import RetrievedChunk, SearchResult
from app.services.service_areas import ServiceAreas

logger = logging.getLogger(__name__)

NO_RESULTS = "No relevant KB content found."
KB_TOOLS = ("search_knowledge_base", "check_service_area")  # the tools whose artifacts are KB chunks


def _json(data: dict) -> str:
    return json.dumps(data)


def make_tools(
    search_fn: Callable[[str], SearchResult],
    areas_fn: Callable[[], ServiceAreas],
    hints_fn: Callable[[], KbHints],
    ops: LeadOps | None = None,
) -> list[BaseTool]:
    """The agent's tools, using the given search, service-area and KB-hint functions (tests pass fakes).

    With `ops` (the leads database), the four lead tools are added.
    """

    @tool(response_format="content_and_artifact")
    def search_knowledge_base(query: str) -> tuple[str, list[dict]]:
        """Search the Vantis knowledge base. Use it only when the context you were given does not answer
        the customer's question. Write the query in the knowledge base's own words."""
        result = search_fn(query)
        if result.no_context or not result.chunks:
            return NO_RESULTS, []
        return format_context(result.chunks), [asdict(c) for c in result.chunks]

    @tool(response_format="content_and_artifact")
    def check_service_area(city: str, state: str | None = None) -> tuple[str, list[dict]]:
        """Check whether a city is in Vantis's listed service areas (KB Section 10). Call it whenever the
        customer names a city, before saying anything about service there."""
        try:
            result = areas_fn().lookup(city, state)
        except Exception as exc:  # DB down or not ingested
            logger.warning("service-area check unavailable: %s", type(exc).__name__)
            return json.dumps({"city": city, "status": "unknown", "note": "service-area list unavailable"}), []
        data = {
            "city": city,
            "status": result.status,  # listed | unverified
            "matched_city": result.matched_city,
            "regions": result.regions,
            "source": result.source,
        }
        summary = hints_fn().summaries.get(SERVICE_AREA_SECTION)
        if not summary:
            return json.dumps(data), []
        return json.dumps(data) + "\n\n" + format_context([RetrievedChunk(**summary)]), [summary]

    kb_tools = [search_knowledge_base, check_service_area]
    if ops is None:
        return kb_tools

    def areas() -> ServiceAreas | None:
        try:
            return ops.areas_fn()
        except Exception:  # the service then marks the location "unknown"
            return None

    @tool(response_format="content_and_artifact")
    def create_lead(state: Annotated[dict, InjectedState]) -> tuple[str, dict]:
        """Save the customer as a sales lead from the details already given in this conversation (takes
        no arguments). Call it when the lead status says it is ready to save; sharing a phone number or
        email counts as wanting follow-up. Never for an existing customer's support issue."""
        lead = state.get("lead", {})
        if state.get("support_case"):
            return _json({"saved": False, "reason": "existing customer's issue: use create_support_request"}), {}
        if state.get("lead_id"):
            note = "already saved; use update_lead for new details"
            return _json({"saved": True, "lead_id": state["lead_id"], "note": note}), {}
        result = service.create_lead(
            ops.engine, lead, session_id=state.get("session_id") or None, llm=ops.reason_llm,
            areas=areas(), reason_now=ops.reason_now,
        )
        ops.calls.append("create")
        if result.lead is None:
            return _json({"saved": False, "missing": result.missing_fields, "invalid": sorted(result.errors)}), {}
        content = {"saved": True, "lead_id": result.lead.id, "fields_saved": sorted(lead)}
        pending = result.lead.reason_source == "template" and not ops.reason_now
        return _json(content), {"kind": "lead", "lead_id": result.lead.id, "saved_fields": dict(lead),
                                "reason_pending": pending}

    @tool(response_format="content_and_artifact")
    def update_lead(state: Annotated[dict, InjectedState]) -> tuple[str, dict]:
        """Save new details the customer gave after their lead was saved (takes no arguments)."""
        lead = state.get("lead", {})
        lead_id = state.get("lead_id")
        if not lead_id:
            missing = ordered_missing(lead, state.get("contact_request", False))
            return _json({"saved": False, "reason": "no lead yet", "missing": missing}), {}
        changes = pending_changes(lead, state.get("lead_saved", {}))
        pending = False
        if changes:
            result = service.update_lead(ops.engine, lead_id, changes, llm=ops.reason_llm, areas=areas(),
                                         reason_now=ops.reason_now)
            ops.calls.append("update")
            pending = result.lead.reason_source == "template" and not ops.reason_now
            content = {"saved": True, "fields_saved": sorted(changes)}
        else:
            content = {"saved": True, "note": "nothing new"}
        return _json(content), {"kind": "lead", "lead_id": lead_id, "saved_fields": dict(lead),
                                "reason_pending": pending}

    @tool(response_format="content_and_artifact")
    def schedule_callback(state: Annotated[dict, InjectedState]) -> tuple[str, dict]:
        """Log the customer's request to be called back, at the time they gave (takes no arguments).
        Needs a saved lead. It is logged as a request: the team confirms it; it is never booked."""
        lead_id, preferred = state.get("lead_id"), state.get("callback_time")
        if not lead_id:
            missing = ordered_missing(state.get("lead", {}), True)
            return _json({"logged": False, "reason": "save the lead first", "missing": missing}), {}
        if not preferred:
            return _json({"logged": False, "missing": ["preferred time"]}), {}
        if state.get("callback_logged") != preferred:  # never log the same request twice
            service.schedule_callback(ops.engine, lead_id, preferred)
            ops.calls.append("callback")
        content = {"logged": True, "status": "requested", "preferred_time": preferred,
                   "note": "the team will confirm it"}
        return _json(content), {"kind": "callback", "status": "requested", "time": preferred}

    @tool(response_format="content_and_artifact")
    def create_support_request(state: Annotated[dict, InjectedState]) -> tuple[str, dict]:
        """Save an existing customer's problem with their system as a support request (takes no
        arguments). Needs their name, phone or email, project details and the issue. Never a sales lead."""
        if state.get("support_request_id"):
            content = {"saved": True, "support_request_id": state["support_request_id"], "note": "already saved"}
            return _json(content), {}
        lead, support = state.get("lead", {}), state.get("support", {})
        missing = support_missing(lead, support)
        if missing:
            return _json({"saved": False, "missing": missing}), {}
        raw = {k: lead[k] for k in ("name", "phone", "email") if lead.get(k)}
        raw |= {"issue": support["issue"], "project_details": support["project_details"]}
        result = service.create_support_request(ops.engine, raw, session_id=state.get("session_id") or None)
        ops.calls.append("support")
        if result.id is None:
            return _json({"saved": False, "missing": result.missing_fields, "invalid": sorted(result.errors)}), {}
        return _json({"saved": True, "support_request_id": result.id}), {"kind": "support", "id": result.id}

    return kb_tools + [create_lead, update_lead, schedule_callback, create_support_request]


def tool_step(message) -> dict:
    """A steps entry for one tool result (a ToolMessage), with a short detail and no message text."""
    name = message.name
    artifact = message.artifact or []
    if name == "check_service_area":
        try:
            data = json.loads(str(message.content).split("\n\n", 1)[0])
            detail = f"{data['city']}: {data['status']}"
        except (ValueError, KeyError):
            detail = "error"
    elif name == "search_knowledge_base":
        sections = sorted({c["section_no"] for c in artifact})
        detail = f"{len(artifact)} chunks" + (f": KB Sections {', '.join(map(str, sections))}" if sections else "")
    else:  # lead tools: what happened, field names only
        detail = _lead_tool_detail(str(message.content))
    if getattr(message, "status", "success") == "error":
        detail = "error"
    return {"node": "tool", "name": name, "detail": detail, "data": {}}


def _lead_tool_detail(content: str) -> str:
    """'saved', 'requested', 'missing: name, phone or email', ... from a lead tool's JSON result."""
    try:
        data = json.loads(content)
    except ValueError:
        return "error"
    if data.get("logged"):
        return "requested"
    if data.get("saved"):
        return "saved" + (f": {', '.join(data['fields_saved'])}" if data.get("fields_saved") else "")
    if data.get("missing"):
        return "missing: " + ", ".join(data["missing"])
    return data.get("reason", "not saved")

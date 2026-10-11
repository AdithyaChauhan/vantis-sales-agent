"""Tools the agent can call (T8a: knowledge search and the service-area check; T8b adds the lead tools).

Each tool returns (content for the LLM, artifact): the artifact holds KB chunk dicts, so the reply's
sources can include what a tool found. The tools return data only; the system prompt says how to word it.
"""
import json
import logging
from dataclasses import asdict
from typing import Callable

from langchain_core.tools import BaseTool, tool

from app.agent.understand import KbHints
from app.rag.answer import format_context
from app.rag.chunker import SERVICE_AREA_SECTION
from app.rag.retriever import RetrievedChunk, SearchResult
from app.services.service_areas import ServiceAreas

logger = logging.getLogger(__name__)

NO_RESULTS = "No relevant KB content found."


def make_tools(
    search_fn: Callable[[str], SearchResult],
    areas_fn: Callable[[], ServiceAreas],
    hints_fn: Callable[[], KbHints],
) -> list[BaseTool]:
    """The agent's tools, using the given search, service-area and KB-hint functions (tests pass fakes)."""

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

    return [search_knowledge_base, check_service_area]


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
    else:
        detail = "done"
    if getattr(message, "status", "success") == "error":
        detail = "error"
    return {"node": "tool", "name": name, "detail": detail, "data": {}}

"""Fakes for agent tests: scripted understand outputs, scripted agent messages, a fake search. No network."""
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import AgentDeps, build_graph
from app.agent.state import turn_input
from app.agent.tools import make_tools
from app.agent.understand import KbHints, Understanding
from app.rag.retriever import RetrievedChunk, SearchResult
from tests.leads_data import AREAS

CATALOG_ROW = RetrievedChunk("s08-plan-a", 8, "Catalog", 3, "table_row", "Offering: Plan A")
FAQ = RetrievedChunk("s15-quote", 15, "FAQ", 5, "faq", "Q: Can I get a quote?\nA: Yes.")
CATALOG_SUMMARY = {
    "chunk_id": "s08-table-summary", "section_no": 8, "section_title": "Catalog", "page": 3,
    "chunk_type": "table_summary",  # the KB Section 8 summary's shape: name; customer; profile (T13e)
    "content": "All rows (Offering; Primary Customer; Reference Profile):\n"
               "- Plan A; Residential; 3 kW rooftop solution\n- Plan B; Residential; 5 kW rooftop solution",
}
AREAS_SUMMARY = {
    "chunk_id": "s10-table-summary", "section_no": 10, "section_title": "Locations", "page": 3,
    "chunk_type": "table_summary", "content": "All rows (Region; Reference Locations):\n- Uttar Pradesh; Lucknow, Kanpur",
}
# KB Section 9 rows (T13a: in the context on a turn that gives the monthly bill)
GUIDANCE = [
    {"chunk_id": "s09-moderate", "section_no": 9, "section_title": "Guidance", "page": 3, "chunk_type": "table_row",
     "content": "Customer Signal: Moderate residential requirement\nPotential Reference Offering: Plan A\n"
                "Do Not Assume: Do not assume 3 kW is suitable without consumption/site information."},
    {"chunk_id": "s09-higher", "section_no": 9, "section_title": "Guidance", "page": 3, "chunk_type": "table_row",
     "content": "Customer Signal: Higher residential requirement\nPotential Reference Offering: Plan B\n"
                "Do Not Assume: Do not assume 5 kW is sufficient for every household."},
]
HINTS = KbHints("KB Section 8: Catalog\nKB Section 10: Locations", {8: CATALOG_SUMMARY, 10: AREAS_SUMMARY}, GUIDANCE)


TOP_LEVEL = ("asked_city", "cities", "callback_time", "issue", "project_details", "declines", "reply_type",
             "ruled_out", "locality", "place_relation", "asks_offerings", "problem_now")


def default_reply_type(intents: list[str], details: dict, top: dict) -> str:
    """What a model would say the message does when a test doesn't say (T13e): details answer, thanks closes, a reply to
    the assistant accepts, a decline declines, anything else asks something new."""
    if top.get("declines"):
        return "declines"
    if details or top.get("callback_time"):
        return "answers"
    if set(intents) <= {"thanks_or_bye"}:
        return "closes"
    if "reply_to_assistant" in intents:
        return "accepts"
    return "asks_new"


def understood(intents, search_query=None, provider="google_genai", **details) -> dict:
    """What the structured-output model returns: raw message, parsed Understanding, no parsing error.

    Keyword arguments are lead details, except the top-level fields in TOP_LEVEL. T13e: reply_type and place_relation
    default to what a model would say (a place named is inside its city)."""
    top = {key: details.pop(key) for key in TOP_LEVEL if key in details}
    top.setdefault("reply_type", default_reply_type(intents, details, top))
    named = details.get("city") or top.get("asked_city") or top.get("cities")
    top.setdefault("place_relation", "inside" if named else "none")
    top.setdefault("asks_offerings", False)
    top.setdefault("problem_now", "yes" if "existing_customer_issue" in intents else "no")
    parsed = Understanding(intents=intents, search_query=search_query, details=details, **top)
    raw = AIMessage(content="", response_metadata={"model_provider": provider})
    return {"raw": raw, "parsed": parsed, "parsing_error": None}


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}],
                     response_metadata={"model_provider": "google_genai"})


def answer(text: str, provider: str = "google_genai") -> AIMessage:
    return AIMessage(content=text, response_metadata={"model_provider": provider})


class Scripted:
    """A fake model: returns (or raises) the scripted outputs in order and records every input."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append(messages)
        output = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(output, Exception):
            raise output
        if isinstance(output, AIMessage):  # a fresh message each call, like a real model
            return output.model_copy(update={"id": None})
        return output


class FakeSearch:
    """Records queries and returns the given result."""

    def __init__(self, result: SearchResult | None = None, error: Exception | None = None):
        self.result = result or SearchResult(chunks=[CATALOG_ROW, FAQ], mode="hybrid")
        self.error = error
        self.queries = []

    def __call__(self, query: str) -> SearchResult:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.result


def make_graph(
    understand_llm, agent_llm, final_llm=None, search=None, small_to_big=False, checkpointer=None, ops=None,
    agent_llm_without_create=None, areas=None,
):
    """The real graph with fakes. Returns (graph, search). ops: the leads database (lead tools on).
    agent_llm_without_create: the model for the rest of a turn after create_lead could not save (T13c).
    areas: the service-area table (default: tests/leads_data.py AREAS)."""
    search = search or FakeSearch()
    areas = areas or AREAS
    tools = make_tools(search, lambda: areas, lambda: HINTS, ops)
    deps = AgentDeps(
        understand_llm=understand_llm,
        agent_llm=agent_llm,
        final_llm=final_llm or Scripted(answer("Final answer without tools.")),
        search_fn=search,
        areas_fn=lambda: areas,
        hints_fn=lambda: HINTS,
        tools=tools,
        checkpointer=checkpointer or InMemorySaver(),
        small_to_big=small_to_big,
        ops=ops,
        agent_llm_without_create=agent_llm_without_create,
    )
    return build_graph(deps), search


def run(graph, message: str, thread: str = "t-1") -> dict:
    return graph.invoke(turn_input(message, thread), {"configurable": {"thread_id": thread}})

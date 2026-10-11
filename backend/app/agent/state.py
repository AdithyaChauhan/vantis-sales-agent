"""The agent's state for one conversation (checkpointed per session_id).

Domain data is kept as plain dicts; LangChain messages are the only objects, because LangGraph's
add_messages reducer needs them.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages


class AgentState(TypedDict, total=False):
    """Persisted across turns: the conversation and the lead/support/callback fields. The rest is reset
    at the start of each turn (turn_input)."""

    # The conversation: customer messages and final replies only (no tool calls)
    messages: Annotated[list[AnyMessage], add_messages]
    # Clean lead details stated so far (LeadData fields): the draft the lead tools save
    lead: dict
    session_id: str
    lead_id: int | None
    lead_saved: dict  # the draft fields last written to the leads table (avoids writing twice)
    callback_status: str | None
    callback_logged: str | None  # the preferred time already logged, so it is not logged twice
    callback_time: str | None  # when the customer wants a call, in their own words
    support_case: bool  # an existing customer's issue: no sales lead for this conversation
    support: dict  # issue and project_details, in the customer's words
    support_request_id: int | None
    asked_cities: list[str]  # cities asked about but not (yet) confirmed as the customer's
    # This turn's tool calls and tool results
    scratch: Annotated[list[AnyMessage], add_messages]
    intents: list[str]
    search_query: str | None
    chunks: list[dict]  # retrieved KB chunks (RetrievedChunk fields)
    no_context: bool
    search_mode: str | None
    validation_errors: dict  # field -> message, for this turn's invalid details
    tool_rounds: int
    steps: list[dict]
    contact_request: bool  # this turn the customer asked to be contacted
    reply: str
    sources: list[dict]
    lead_summary: dict | None
    reason_pending: int | None  # a lead whose reason the API writes after replying
    degraded: bool


def turn_input(message: str, session_id: str = "") -> dict:
    """The graph input for one customer message: the message plus a reset of every per-turn field."""
    return {
        "messages": [HumanMessage(message)],
        "session_id": session_id,
        "contact_request": False,
        "lead_summary": None,
        "reason_pending": None,
        "scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        "intents": [],
        "search_query": None,
        "chunks": [],
        "no_context": False,
        "search_mode": None,
        "validation_errors": {},
        "tool_rounds": 0,
        "steps": [],
        "reply": "",
        "sources": [],
        "degraded": False,
    }

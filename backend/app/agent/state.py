"""The agent's state for one conversation (checkpointed per session_id).

Domain data is kept as plain dicts; LangChain messages are the only objects, because LangGraph's
add_messages reducer needs them.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages


class AgentState(TypedDict, total=False):
    """Persisted across turns: messages and lead. Everything else is reset at the start of each turn."""

    # The conversation: customer messages and final replies only (no tool calls)
    messages: Annotated[list[AnyMessage], add_messages]
    # Clean lead details stated so far (LeadData fields); T8b saves them
    lead: dict
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
    reply: str
    sources: list[dict]
    degraded: bool


def turn_input(message: str) -> dict:
    """The graph input for one customer message: the message plus a reset of every per-turn field."""
    return {
        "messages": [HumanMessage(message)],
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

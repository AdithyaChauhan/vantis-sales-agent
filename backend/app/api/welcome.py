"""GET /api/welcome: the chat page's welcome text and starter questions (no LLM call).

The agent replies with the same text to a plain hello (app/agent/welcome.py holds it once).
"""
from fastapi import APIRouter

from app.agent.welcome import STARTER_QUESTIONS, WELCOME_TEXT

router = APIRouter(tags=["chat"])


@router.get("/api/welcome")
def get_welcome() -> dict:
    """What the assistant can help with, its limits, and a few starter questions."""
    return {"text": WELCOME_TEXT, "starter_questions": STARTER_QUESTIONS}

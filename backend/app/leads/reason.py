"""The one-sentence reason for a lead's status, written by the fast LLM from the score breakdown.

The LLM sees only the status, score, rule labels, caps and city: no name, phone, email or money amount.
Any doubtful reply is replaced by a template sentence built from the same breakdown.
"""
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from app.core.llm import LLMError, invoke_llm
from app.core.prompts import load_prompt

logger = logging.getLogger(__name__)

MAX_REASON_CHARS = 200
STATUSES = ("HOT", "WARM", "COLD")
# KB Section 16 topics a reason must never touch. Whole words or word starts, so "premium" is not "EMI".
FORBIDDEN_RE = re.compile(
    r"₹|\brs\b|\binr\b|\bsubsid|\bemis?\b|\bsaving|\bfinanc|\bguarantee|\bpric", re.IGNORECASE
)
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def build_prompt(breakdown: dict, city: str | None) -> str:
    """The prompt text: status, score, city and the breakdown labels only."""
    factors = "\n".join(f"- {r['label']}: {r['points']} of {r['max']}" for r in breakdown["rules"])
    caps = "\n".join(f"- {c['label']}" for c in breakdown["caps"]) or "- none"
    filled = {
        "{status}": breakdown["lead_status"],
        "{score}": str(breakdown["lead_score"]),
        "{city}": city or "not given",
        "{factors}": factors,
        "{caps}": caps,
    }
    text = load_prompt("lead_qualification_prompt")
    for placeholder, value in filled.items():
        text = text.replace(placeholder, value)
    return text


def template_reason(breakdown: dict) -> str:
    """Fallback reason from the breakdown, e.g. 'HOT: ready to buy, clear requirement: 5 kW system, ...'."""
    earned = [r["label"] for r in breakdown["rules"] if r["points"] > 0]
    text = f"{breakdown['lead_status']}: " + (", ".join(_lower_first(x) for x in earned) or "few details so far")
    if breakdown["caps"]:
        text += "; " + "; ".join(_lower_first(c["label"]) for c in breakdown["caps"])
    return text + "."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def problem_with(reply: str, status: str) -> str | None:
    """Why an LLM reply can't be used as the reason, or None if it is fine."""
    if not reply:
        return "empty"
    if len(reply) > MAX_REASON_CHARS or "\n" in reply:
        return "too long"
    if len(_SENTENCE_END_RE.findall(reply)) > 1:
        return "more than one sentence"
    others = [s for s in STATUSES if s != status]
    if re.search(rf"\b({'|'.join(others)})\b", reply, re.IGNORECASE):
        return "names another status"
    if FORBIDDEN_RE.search(reply):
        return "forbidden topic"
    return None


def write_reason(breakdown: dict, city: str | None, llm: Runnable) -> tuple[str, str]:
    """(reason, source): source is 'llm', or 'template' when the LLM failed or its reply was unusable."""
    try:
        reply = invoke_llm(llm, [HumanMessage(build_prompt(breakdown, city))]).text
    except LLMError:
        reply = ""
    reply = reply.strip().strip('"').strip()
    problem = problem_with(reply, breakdown["lead_status"])
    if problem:
        logger.info("lead reason: template used (%s)", problem)
        return template_reason(breakdown), "template"
    return reply, "llm"

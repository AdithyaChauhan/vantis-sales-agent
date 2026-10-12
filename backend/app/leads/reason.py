"""The one-sentence reason for a lead's status, written by the LLM (app: reason_llm, the main model) from the
score breakdown.

The LLM sees only the status, score, rule labels, caps, city and what keeps the lead from the next status:
no name, phone, email or money amount. Any doubtful reply is replaced by a template sentence built from the
same breakdown.
"""
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from app.core.llm import LLMError, invoke_llm
from app.core.prompts import load_prompt
from app.leads.scoring import next_status_gap

logger = logging.getLogger(__name__)

MAX_REASON_CHARS = 240  # about 30 words, with the "not HOT yet: ..." ending (T12a)
STATUSES = ("HOT", "WARM", "COLD")
# KB Section 16 topics a reason must never touch. Whole words or word starts, so "premium" is not "EMI".
FORBIDDEN_RE = re.compile(
    r"₹|\brs\b|\binr\b|\bsubsid|\bemis?\b|\bsaving|\bfinanc|\bguarantee|\bpric", re.IGNORECASE
)
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def gap_phrase(breakdown: dict) -> str | None:
    """'not HOT yet: no timeline' for a WARM lead (T12a), or None for a HOT lead."""
    gap = next_status_gap(breakdown)
    if gap is None:
        return None
    next_status, gaps = gap
    return f"not {next_status} yet: " + (", ".join(gaps) or "more details needed")


def build_prompt(breakdown: dict, city: str | None) -> str:
    """The prompt text: status, score, city, the breakdown labels and the next-status gap only."""
    factors = "\n".join(f"- {r['label']}: {r['points']} of {r['max']}" for r in breakdown["rules"])
    caps = "\n".join(f"- {c['label']}" for c in breakdown["caps"]) or "- none"
    filled = {
        "{status}": breakdown["lead_status"],
        "{score}": str(breakdown["lead_score"]),
        "{city}": city or "not given",
        "{factors}": factors,
        "{caps}": caps,
        "{next_status}": gap_phrase(breakdown) or "none: it is already HOT",
    }
    text = load_prompt("lead_qualification_prompt")
    for placeholder, value in filled.items():
        text = text.replace(placeholder, value)
    return text


def template_reason(breakdown: dict) -> str:
    """Fallback reason from the breakdown, e.g. 'WARM: interested in options, ...; not HOT yet: no timeline.'

    The gap names a cap's cause too, so a capped lead's reason says why it is not HOT.
    """
    earned = [r["label"] for r in breakdown["rules"] if r["points"] > 0]
    text = f"{breakdown['lead_status']}: " + (", ".join(_lower_first(x) for x in earned) or "few details so far")
    gap = gap_phrase(breakdown)
    if gap:
        text += "; " + gap
    return text + "."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def problem_with(reply: str, status: str, next_status: str | None = None) -> str | None:
    """Why an LLM reply can't be used as the reason, or None if it is fine.

    next_status (T12a): for a lead that is not HOT, the reply must say "not <next_status> yet"; that phrase is
    the only place another status may appear.
    """
    if not reply:
        return "empty"
    if len(reply) > MAX_REASON_CHARS or "\n" in reply:
        return "too long"
    if len(_SENTENCE_END_RE.findall(reply)) > 1:
        return "more than one sentence"
    checked = reply
    if next_status:
        phrase = re.compile(rf"\bnot {next_status} yet\b", re.IGNORECASE)
        if not phrase.search(reply):
            return "does not say what is missing for the next status"
        checked = phrase.sub("", reply)
    others = [s for s in STATUSES if s != status]
    if re.search(rf"\b({'|'.join(others)})\b", checked, re.IGNORECASE):
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
    gap = next_status_gap(breakdown)
    problem = problem_with(reply, breakdown["lead_status"], gap[0] if gap else None)
    if problem:
        logger.info("lead reason: template used (%s)", problem)
        return template_reason(breakdown), "template"
    return reply, "llm"

"""The one-sentence reason for a lead's status, written by the LLM (app: reason_llm, the main model) from the
score breakdown.

The LLM sees only the status, score, rule labels, caps, city and what keeps the lead from the next status:
no name, phone, email or money amount. Any doubtful reply is replaced by a template sentence built from the
same breakdown.

T12b: the LLM writes only the first part ("why it is rated WARM"); code adds the ending "; not HOT yet: …" once.
In the T12a seed the LLM copied the prompt's label ("is missing for the next status: not HOT yet: …") or said the
same thing twice ("only exploring, not WARM yet: only exploring").
"""
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from app.core.llm import LLMError, invoke_llm
from app.core.prompts import load_prompt
from app.leads.scoring import status_gaps

logger = logging.getLogger(__name__)

MAX_REASON_CHARS = 240  # about 30 words, with the "not HOT yet: ..." ending (T12a)
STATUSES = ("HOT", "WARM", "COLD")
# KB Section 16 topics a reason must never touch. Whole words or word starts, so "premium" is not "EMI".
FORBIDDEN_RE = re.compile(
    r"₹|\brs\b|\binr\b|\bsubsid|\bemis?\b|\bsaving|\bfinanc|\bguarantee|\bpric", re.IGNORECASE
)
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
# The prompt's own words, which a reply must never copy (T12b: Sana Khan's seeded reason did)
_INSTRUCTION_RE = re.compile(r"\bnext status\b|\bmissing for\b|\bthe ending\b", re.IGNORECASE)
# An ending the LLM wrote anyway after a separator ("…, not HOT yet: no timeline"): cut off, code writes it once.
# Inside a clause ("…and they are not WARM yet: …", T12b seed) it is not cut, so the part names another status and
# the template is used: cutting it there left "…and they are;".
_OWN_ENDING_RE = re.compile(r"(?:\s*[,;:–-]\s*|\s+but\s+)(?:but\s+)?not (?:HOT|WARM) yet\b.*$",
                            re.IGNORECASE | re.DOTALL)
_STOP_WORDS = {"city", "with", "the", "and", "but", "not", "yet"}
# A part that stops mid-clause once a written-out ending is cut ("…in words only, and"; T12b seed, second run)
_DANGLING_RE = re.compile(r"\b(?:and|but|or|so|with|because|as|of|to|for|the|a|an|is|are|was|were|they|who|which|"
                          r"that|while|although)$", re.IGNORECASE)


def gap_phrase(breakdown: dict) -> str | None:
    """'not HOT yet: no timeline' for a WARM lead (T12a), or None for a HOT lead."""
    gap = status_gaps(breakdown)
    if gap is None:
        return None
    next_status, gaps = gap
    return f"not {next_status} yet: " + (", ".join(label for _, label in gaps) or "more details needed")


def build_prompt(breakdown: dict, city: str | None) -> str:
    """The prompt text: status, score, city, the breakdown labels and the ending code will add."""
    factors = "\n".join(f"- {r['label']}: {r['points']} of {r['max']}" for r in breakdown["rules"])
    caps = "\n".join(f"- {c['label']}" for c in breakdown["caps"]) or "- none"
    filled = {
        "{status}": breakdown["lead_status"],
        "{score}": str(breakdown["lead_score"]),
        "{city}": city or "not given",
        "{factors}": factors,
        "{caps}": caps,
        "{ending}": f'"; {gap_phrase(breakdown)}."' if gap_phrase(breakdown) else "none",
    }
    text = load_prompt("lead_qualification_prompt")
    for placeholder, value in filled.items():
        text = text.replace(placeholder, value)
    return text


def template_reason(breakdown: dict) -> str:
    """Fallback reason from the breakdown, e.g. 'WARM: interested in options, ...; not HOT yet: no timeline.'

    The factors leave out the rules the ending already names (T12b), so nothing is said twice; the ending names a
    cap's cause too, so a capped lead's reason says why it is not HOT.
    """
    gap = status_gaps(breakdown)
    named = {rule for rule, _ in gap[1]} if gap else set()
    earned = [r["label"] for r in breakdown["rules"] if r["points"] > 0 and r["rule"] not in named]
    text = f"{breakdown['lead_status']}: " + (", ".join(_lower_first(x) for x in earned) or "few details so far")
    ending = gap_phrase(breakdown)
    if ending:
        text += "; " + ending
    return text + "."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _repeats(text: str, label: str) -> bool:
    """True if the text already says what a gap label says: every content word of the label is in it."""
    words = {w for w in re.findall(r"[a-z0-9-]+", label.lower()) if len(w) > 3 and w not in _STOP_WORDS}
    found = set(re.findall(r"[a-z0-9-]+", text.lower()))
    return bool(words) and words <= found


def problem_with(part: str, status: str, gap_labels: list[str] | None = None) -> str | None:
    """Why the LLM's part of the reason can't be used, or None if it is fine.

    The part is the sentence before the code-made ending, so no other status may appear in it at all (T12b), it
    must not copy the prompt's wording, and it must not repeat what the ending will say.
    """
    if not part:
        return "empty"
    if len(part) > MAX_REASON_CHARS or "\n" in part:
        return "too long"
    if _SENTENCE_END_RE.search(part):  # the part's own full stop is already cut off, so any other ends a sentence
        return "more than one sentence"
    if _INSTRUCTION_RE.search(part):
        return "copies the prompt's wording"
    if _DANGLING_RE.search(part.rstrip(" ,;:")):
        return "stops mid-clause"
    others = [s for s in STATUSES if s != status]
    if re.search(rf"\b({'|'.join(others)})\b", part, re.IGNORECASE):
        return "names another status"
    if FORBIDDEN_RE.search(part):
        return "forbidden topic"
    if any(_repeats(part, label) for label in gap_labels or []):
        return "repeats the ending"
    return None


def write_reason(breakdown: dict, city: str | None, llm: Runnable) -> tuple[str, str]:
    """(reason, source): source is 'llm', or 'template' when the LLM failed or its reply was unusable."""
    try:
        reply = invoke_llm(llm, [HumanMessage(build_prompt(breakdown, city))]).text
    except LLMError:
        reply = ""
    part = _OWN_ENDING_RE.sub("", reply.strip().strip('"').strip()).strip().rstrip(".;,:").strip()
    gap = status_gaps(breakdown)
    ending = gap_phrase(breakdown)
    problem = problem_with(part, breakdown["lead_status"], [label for _, label in gap[1]] if gap else None)
    reason = f"{part}; {ending}." if ending else f"{part}."
    if not problem and len(reason) > MAX_REASON_CHARS:
        problem = "too long"
    if problem:
        logger.info("lead reason: template used (%s)", problem)
        return template_reason(breakdown), "template"
    return reason, "llm"

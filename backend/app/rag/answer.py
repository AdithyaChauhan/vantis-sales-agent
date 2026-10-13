"""Helpers for grounded answers: the context block, citations, sources and the retrieve step.

The agent (app/agent/graph.py) uses them; nothing here calls an LLM or the database.
"""
import logging
import re

from langchain_core.messages import HumanMessage

from app.core.prompts import load_prompt
from app.rag.retriever import RetrievedChunk, SearchResult

logger = logging.getLogger(__name__)

LLM_DOWN_REPLY = "I'm having trouble right now. Please try again in a minute."
SNIPPET_CHARS = 160
_CITATION_RE = re.compile(r"KB Section (\d+)")
_LABEL_RE = re.compile(r"([ \t]*)\[KB Section (\d+)(?::[^\[\]]*)?\]")  # spaces before it, section number
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # "4.5 kW" and "₹2,000" are not split
# A claim that something was saved or logged: "your details are saved", "I've recorded your bill". Future or
# conditional wording ("will be saved", "once you share it, I can save it") is not a claim (T12a).
_SAVED = r"(?:saved|recorded|stored|registered|logged)"
# Handing on is a claim only when the sentence names who got it: "I've passed your details to the team"
# (but "I've shared the steps above" is not).
_HANDED_ON = r"(?:passed|forwarded|sent|shared)"
_DONE_BY = r"\b(?:i|we)(?:'ve|’ve| have)?\s+(?:now\s+|just\s+|also\s+|already\s+)?"
_DONE_TO = r"\b(?:is|are|was|were|has been|have been)\s+(?:now\s+|all\s+|already\s+|successfully\s+)?"
_SAVED_RE = re.compile(rf"{_DONE_BY}{_SAVED}\b|{_DONE_TO}{_SAVED}\b|^\s*(?:all\s+)?{_SAVED}\b", re.IGNORECASE)
_HANDED_ON_RE = re.compile(rf"{_DONE_BY}{_HANDED_ON}\b|{_DONE_TO}{_HANDED_ON}\b", re.IGNORECASE)
_TEAM_RE = re.compile(r"\b(?:team|sales|someone|representative|colleague|expert)s?\b", re.IGNORECASE)
# A thank-you is never a KB fact ("Thank you for your interest ... [KB Section 2]" in the T12a replay)
_THANKS_RE = re.compile(r"^\W*(?:thanks|thank you)\b", re.IGNORECASE)
# "The team will follow up" promises a human action that has not happened (KB Section 16); the assistant may
# only say the team can (T8b rule, slipped on Groq in the T12a replay)
_PROMISE_RE = re.compile(
    r"(\s+will|'ll|’ll)(\s+(?:follow up|call|contact|reach out|get in touch|get back))\b", re.IGNORECASE
)
# KB Section 10's own words ("Reference Locations", "Reference service area") are office terms; the customer reads
# "service areas" (T12b). Only text outside [KB Section ...] tags is changed, so the label stays exact.
_TAG_OR_TEXT_RE = re.compile(r"(\[KB Section \d+(?::[^\[\]]*)?\])|([^\[]+|\[)")
_REFERENCE_AREA_RE = re.compile(r"\breference\s+(?:service\s+)?(?:locations?|areas?)\b", re.IGNORECASE)
# T13c: topics the KB never covers. A reply states no policy on them either way ("Vantis doesn't offer a specific
# discount rate" answered "can I get 20% discount?"); it says it can't confirm and offers the sales team
_UNCOVERED_RE = re.compile(r"\b(?:discounts?|coupons?|cashback|promo(?:tion)?s?|brands?|"
                           r"(?:special|current|seasonal|festive|festival|ongoing)\s+offers?)\b", re.IGNORECASE)
_CANT_CONFIRM_RE = re.compile(r"\b(?:can(?:'|’)?t|cannot|can not|unable to|not able to)\s+confirm\b|"
                              r"\b(?:don(?:'|’)?t|do not)\s+have\s+(?:any\s+)?(?:information|details)\b|"
                              r"\bnot sure\b", re.IGNORECASE)


def citation_label(chunk: RetrievedChunk) -> str:
    """The label the model must cite, e.g. [KB Section 8: Product & Service Catalog]."""
    return f"[KB Section {chunk.section_no}: {chunk.section_title}]"


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Each chunk as its label on one line and its text below, separated by blank lines."""
    return "\n\n".join(f"{citation_label(c)}\n{c.content}" for c in chunks)


def rag_message(
    question: str, chunks: list[RetrievedChunk], lead_progress: str = "", area_checks: str = "",
    package_guidance: str = "",
) -> HumanMessage:
    """The customer's message wrapped in the RAG prompt with the context (empty when nothing was found),
    (T13b) the package rule of a bill turn right after it (its line goes on other turns), the lead progress block
    (what is known and still needed; "(not tracked)" without the lead tools) and (T12b) the service-area checks
    the code made for the cities named ("(no city named)" when none).

    The customer's text is filled in last, so a "{context}" typed by the customer is left as it is.
    """
    text = load_prompt("rag_prompt").replace("{context}", format_context(chunks))
    text = text.replace("{package_guidance}\n\n", f"{package_guidance}\n\n" if package_guidance else "")
    text = text.replace("{lead_progress}", lead_progress or "(not tracked)")
    text = text.replace("{service_area_check}", area_checks or "(no city named)")
    return HumanMessage(text.replace("{question}", question))


def normalize_citations(reply: str) -> str:
    """Turn full-width brackets 【 】 into [ ] (gpt-oss on Groq sometimes cites with them)."""
    return reply.replace("【", "[").replace("】", "]")


def strip_tool_citations(reply: str, tool_names: list[str]) -> str:
    """Remove a tool cited like a source, e.g. "[schedule_callback result]" (only KB sections are sources).

    The system prompt forbids it; this is the safety net, and it logs a warning so a slipping prompt shows.
    """
    words = "|".join(re.escape(w) for w in ["result", *tool_names] if w)
    pattern = re.compile(rf"\s*\[(?!KB Section)[^\[\]]*\b(?:{words})\b[^\[\]]*\]", re.IGNORECASE)
    cleaned = pattern.sub("", reply)
    if cleaned != reply:
        logger.warning("reply cited a tool; removed")
    return cleaned


def split_sentences(line: str) -> list[str]:
    """One line of a reply as sentences; a piece holding only KB tags belongs to the sentence before it."""
    return _SENTENCE_SPLIT_RE.split(line)


def without_tags(text: str) -> str:
    """The text with its [KB Section N: Title] tags removed."""
    return _LABEL_RE.sub("", text).strip()


def is_claim(sentence: str) -> bool:
    """True if the sentence says something was saved, recorded or logged, or passed to the team (T12a)."""
    text = without_tags(sentence)
    return bool(_SAVED_RE.search(text) or (_HANDED_ON_RE.search(text) and _TEAM_RE.search(text)))


def strip_unsourced_citations(reply: str, chunks: list[RetrievedChunk]) -> tuple[str, int]:
    """Remove the KB tags this turn's sources don't support (T12a): a section that is not among this turn's
    chunks (every tag when there are none, e.g. a validation message), and any tag on a question, a thank-you or
    a "saved" or "logged" sentence, which are never KB facts. Returns (reply, tags removed).

    The prompts ask for this; this is the safety net, and it logs a warning so a slipping prompt shows.
    """
    sections = {c.section_no for c in chunks}
    removed = 0

    def keep_tags(text: str, facts: bool) -> str:
        nonlocal removed

        def one(match: re.Match) -> str:
            nonlocal removed
            if facts and int(match.group(2)) in sections:
                return match.group(0)
            removed += 1
            return ""

        return _LABEL_RE.sub(one, text)

    lines = []
    for line in reply.split("\n"):
        kept, facts = [], True
        for piece in split_sentences(line):
            bare = without_tags(piece)
            if bare:  # a sentence; tags alone take the verdict of the sentence before them
                facts = not (bare.endswith("?") or is_claim(bare) or _THANKS_RE.match(bare))
            piece = keep_tags(piece, facts).strip()
            if piece:
                kept.append(piece)
        indent = re.match(r"[ \t]*", line).group(0)  # keeps a list item's indentation
        lines.append(indent + " ".join(kept) if kept else "")
    if not removed:
        return reply, 0
    logger.warning("reply cited KB sections without support from this turn; removed", extra={"removed": removed})
    return "\n".join(lines).strip(), removed


def soften_promises(reply: str) -> tuple[str, int]:
    """'our team will follow up' becomes 'our team can follow up' (T12a): a reply never promises that a person
    will call, contact or follow up. Returns (reply, changes). The system prompt asks for this; this is the net."""
    changed, count = _PROMISE_RE.subn(lambda m: " can" + m.group(2), reply)
    if count:
        logger.warning("reply promised a human action; softened", extra={"changed": count})
    return changed, count


def _uncovered_topics(sentence: str, chunks: list[RetrievedChunk]) -> list[str]:
    """The uncovered-topic words a sentence uses that no chunk of this turn mentions ("discount" with no KB text on
    discounts)."""
    found = []
    for match in _UNCOVERED_RE.finditer(sentence):
        word = r"\s+".join(re.escape(part) for part in match.group(0).lower().rstrip("s").split())
        if not any(re.search(rf"\b{word}s?\b", c.content, re.IGNORECASE) for c in chunks):
            found.append(match.group(0))
    return found


def remove_uncovered_policies(reply: str, chunks: list[RetrievedChunk], honest: str) -> tuple[str, list[str]]:
    """A sentence that states something about discounts, coupons, cashback, promotions, special offers or brands,
    when this turn's KB context never mentions that word, is replaced by `honest` (one line saying it can't be
    confirmed, with the sales team; T13c). Questions and sentences that already say it can't be confirmed stay.
    Returns (reply, the removed sentences). The system prompt asks for this; this is the net."""
    lines, removed, replaced = [], [], False
    for line in reply.split("\n"):
        kept, dropped_last = [], False
        for piece in split_sentences(line):
            text = without_tags(piece)
            if not text:  # tags alone follow the sentence before them
                if piece.strip() and not dropped_last:
                    kept.append(piece.strip())
                continue
            dropped_last = (not text.endswith("?") and not _CANT_CONFIRM_RE.search(text)
                            and bool(_uncovered_topics(text, chunks)))
            if dropped_last:
                removed.append(text)
                if not replaced:
                    kept.append(honest)
                    replaced = True
            elif piece.strip():
                kept.append(piece.strip())
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    if not removed:
        return reply, []
    logger.warning("reply stated a policy the KB doesn't cover; replaced", extra={"removed": len(removed)})
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), removed


def plain_service_area_words(reply: str) -> tuple[str, int]:
    """'reference locations' and 'reference service area(s)' become 'service area(s)' outside KB tags (T12b).
    Returns (reply, changes). The system prompt asks for this; this is the net."""
    count = 0

    def plain(match: re.Match) -> str:
        nonlocal count
        count += 1
        word = match.group(0)
        plural = word.lower().endswith("s")
        text = "service areas" if plural else "service area"
        return text[0].upper() + text[1:] if word[0].isupper() else text

    def part(match: re.Match) -> str:
        return match.group(1) or _REFERENCE_AREA_RE.sub(plain, match.group(2))

    changed = _TAG_OR_TEXT_RE.sub(part, reply)
    if count:
        logger.warning("reply used KB office words for service areas; replaced", extra={"changed": count})
    return changed, count


def tidy_citations(reply: str, chunks: list[RetrievedChunk]) -> str:
    """Each KB section cited once, with its full label (the prompt asks for this; this is the safety net).

    Only a section's last citation is kept (after a list, that is the end of the list), and a bare
    "[KB Section 10]" gets its title from the chunks so the chat page can show it as a tag.
    """
    titles = {c.section_no: c.section_title for c in chunks}
    remaining: dict[int, int] = {}
    for match in _LABEL_RE.finditer(reply):
        remaining[int(match.group(2))] = remaining.get(int(match.group(2)), 0) + 1

    def keep_last(match: re.Match) -> str:
        section = int(match.group(2))
        remaining[section] -= 1
        if remaining[section]:
            return ""  # a later citation of the same section stays
        title = titles.get(section)
        return match.group(1) + (f"[KB Section {section}: {title}]" if title else match.group(0).lstrip(" \t"))

    tidied = _LABEL_RE.sub(keep_last, reply)
    if tidied != reply:
        logger.info("citations tidied")
    return tidied


def cited_sections(reply: str) -> list[int]:
    """KB Section numbers the reply cites, in order of first mention, without repeats."""
    return list(dict.fromkeys(int(n) for n in _CITATION_RE.findall(reply)))


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The chunk text on one line, cut at a word boundary after at most `limit` characters."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"


def pick_sources(reply: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """One source per KB Section, because citations are per section.

    Uses the context chunks whose section the reply cites, in retrieval order; a reply that cites none (a
    question, a confirmation) has no sources (T12a: it used to get every context section, so a question showed
    a "KB 10" source). A section's first (highest-ranked) chunk gives the snippet.
    """
    cited = set(cited_sections(reply))
    used = [c for c in chunks if c.section_no in cited]
    if cited - {c.section_no for c in chunks}:
        logger.warning("reply cites a section that was not in the context")
    first_per_section = {}
    for c in used:
        first_per_section.setdefault(c.section_no, c)
    return [
        {
            "section": str(c.section_no),
            "title": c.section_title,
            "page": c.page,
            "snippet": snippet(c.content),
        }
        for c in first_per_section.values()
    ]


def retrieve_step(result: SearchResult) -> dict:
    """The retrieve entry of `steps`."""
    detail = f"{len(result.chunks)} chunks ({result.mode})"
    if result.no_context:
        detail = f"no relevant KB context ({result.mode})"
    return {
        "node": "retrieve",
        "detail": detail,
        "data": {"mode": result.mode, "chunks": len(result.chunks), "no_context": result.no_context},
    }

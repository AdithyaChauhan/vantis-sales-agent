"""The safety net (T13d): the reply when no model answered a turn (every answer model failed or was paused, and the
last try too). It replaces "I'm having trouble right now", which on Google's overloaded days was most of the replies.

It is built only from things that need no model, so it can't invent anything:
1. the code's service-area checks of the cities named (fixed sentences, with the KB Section 10 tag);
2. one or two KB sentences from this turn's chunks that share a key word with the message: an FAQ answer first, never a
   sentence that tells the assistant what to do, each section's tag after its sentences;
3. what this turn saved (the tools' or the code's own save), with its question;
4. otherwise the next missing detail as the one question, or for a customer who has only asked questions a fixed
   offer (once per chat, never after a skip or a no), else "Anything else you'd like to know?".
When no sentence matches, it says plainly that the information doesn't cover it; a reply is never only a question.
"""
import re

from app.agent import welcome
from app.agent.leads import _wanted_item
from app.agent.packages import SIZE_RE
from app.agent.text_details import asks_what_is_offered
from app.rag.answer import citation_label, split_sentences
from app.rag.keywords import key_words
from app.rag.chunker import CATALOG_SECTION, PACKAGE_GUIDANCE_SECTION, SERVICE_AREA_SECTION
from app.rag.retriever import RetrievedChunk

MAX_SENTENCES = 2
MIN_WORDS = 5  # shorter prose lines are headings or bullet fragments ("Residential", "• Full name")
MAX_LIST_ROWS = 8
PACKAGE_SECTIONS = {CATALOG_SECTION, PACKAGE_GUIDANCE_SECTION}  # one package named alone only for a settled size
# A sentence that tells the assistant what to do, never a fact for the customer: "should" or "must" anywhere, the AI or
# the document named, or an order at the start (or after a short opening phrase: "For sizing questions, collect…")
_INSTRUCTION_RE = re.compile(
    r"\b(?:should|must)\b|\bthe\s+ai\b|\bai\s+response\b|\bthis\s+document\b|\bknowledge\s+base\b|"
    r"\bfor\s+this\s+assessment\b|^(?:[^,]{0,60},\s*)?(?:explain|collect|ask|offer|say|tell|use|avoid|treat|create|"
    r"never|do\s+not|don['’]t|ensure|mention|recommend|share|communicate|prioritize|prioritise)\b", re.I)


def is_instruction(sentence: str) -> bool:
    """True for a KB sentence written for the assistant ("Explain that a consultation is required…")."""
    return bool(_INSTRUCTION_RE.search(sentence.strip()))


def _fields(line: str) -> list[tuple[str, str]]:
    key, _, value = line.partition(":")
    return [(key.strip(), value.strip())] if value.strip() else []


def _row_text(chunk: dict, size: str | None) -> str | None:
    """A table row as "<first value>: <the most useful other value>" (with a settled size, the value naming it)."""
    fields = [f for line in str(chunk["content"]).split("\n") for f in _fields(line)]
    if len(fields) < 2:
        return None
    first, others = fields[0][1], [value for _, value in fields[1:] if not is_instruction(value)]
    if size:
        others = [v for v in others if re.search(rf"\b{re.escape(size)}\s*kw\b", v, re.I)] or others
    return _sentence(f"{first}: {others[0]}") if others else None


def _sentence(text: str) -> str:
    """The text with a full stop at the end, unless it already ends a sentence."""
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _summary_rows(chunk: dict, size: str | None) -> list[str]:
    """A summary chunk's rows as "first (second)" (with a settled size, only the row naming it, as "first: value")."""
    rows = [line[2:].split("; ") for line in str(chunk["content"]).split("\n") if line.startswith("- ")]
    if size:
        sized = [row for row in rows if any(re.search(rf"\b{re.escape(size)}\s*kw\b", v, re.I) for v in row)]
        return [_sized_value(row, size) for row in sized[:1]]
    return [f"{row[0]} ({row[1]})" if len(row) > 1 else row[0] for row in rows[:MAX_LIST_ROWS]]


def _sized_value(row: list[str], size: str) -> str:
    """A summary row for a settled size: "<name>: <the value naming the size>", or the row's name when nothing names
    it."""
    named = next((v for v in row[1:] if re.search(rf"\b{re.escape(size)}\s*kw\b", v, re.I)), None)
    return _sentence(f"{row[0]}: {named}" if named else row[0])


def candidates(chunk: dict, size: str | None) -> list[tuple[str, str]]:
    """(text to quote, text to match) pairs a chunk can give, by its type. Instructions are never quoted."""
    kind, content = chunk["chunk_type"], str(chunk["content"])
    if kind == "faq":
        _, _, answer = content.partition("\nA:")
        return [(s.strip(), content) for s in split_sentences(answer.strip()) if s.strip() and not is_instruction(s)]
    if kind == "table_summary":
        rows = _summary_rows(chunk, size)
        if not rows or (chunk["section_no"] in PACKAGE_SECTIONS and not size and len(rows) < 2):
            return []
        text = rows[0] if len(rows) == 1 else "\n".join(f"{i}. {row}" for i, row in enumerate(rows, start=1))
        return [(text, content)]
    if kind == "table_row":
        if chunk["section_no"] in PACKAGE_SECTIONS and not (size and re.search(rf"\b{re.escape(size)}\s*kw\b", content,
                                                                               re.I)):
            return []  # one package named alone only for the size the customer settled (all-or-none rule)
        text = _row_text(chunk, size)
        return [(text, content)] if text else []
    found = []
    for line in content.split("\n"):
        if line.lstrip().startswith("•"):
            continue
        for sentence in split_sentences(line.strip()):
            if len(sentence.split()) >= MIN_WORDS and not is_instruction(sentence):
                found.append((sentence.strip(), sentence))
    return found


def kb_sentences(chunks: list[dict], question: str, size: str | None, offer: bool = False
                 ) -> list[tuple[str, dict]]:
    """Up to MAX_SENTENCES (text, chunk) that share a key word with the question: the package row for a settled size
    first, then the most shared words with an FAQ answer counting one more and winning a tie (Adithya: prefer the FAQ
    when one matches), then the retrieval order; a second one only from the same chunk or with 2+ shared words, so a
    better-matching sentence still follows the FAQ. `offer`: the message asks what Vantis offers, so the KB Section 8
    list of every offering comes first."""
    wanted = key_words(question)
    scored = []
    for rank, chunk in enumerate(chunks):
        for order, (text, match_text) in enumerate(candidates(chunk, size)):
            shared = len(wanted & key_words(match_text))
            sized = bool(size and chunk["section_no"] in PACKAGE_SECTIONS) or (
                offer and chunk["section_no"] == CATALOG_SECTION and chunk["chunk_type"] == "table_summary")
            if shared or sized:
                faq = 1 if chunk["chunk_type"] == "faq" else 0
                scored.append(((sized, shared + faq, faq, -rank, -order), text, chunk, shared))
    scored.sort(key=lambda item: item[0], reverse=True)
    picked: list[tuple[str, dict]] = []
    for _, text, chunk, shared in scored:
        if len(picked) == MAX_SENTENCES:
            break
        if picked and not (chunk["chunk_id"] == picked[0][1]["chunk_id"] or shared >= 2):
            continue
        if text not in (t for t, _ in picked):
            picked.append((text, chunk))
    return picked


def _tagged(pairs: list[tuple[str, dict]]) -> str:
    """The sentences in order, each section's tag after its last sentence (a list keeps its own lines)."""
    parts = []
    for i, (text, chunk) in enumerate(pairs):
        last_of_section = all(c["section_no"] != chunk["section_no"] for _, c in pairs[i + 1:])
        tag = f" {citation_label(RetrievedChunk(**chunk))}" if last_of_section else ""
        parts.append(text + tag)
    return " ".join(parts) if all("\n" not in p for p in parts) else "\n\n".join(parts)


def city_lines(area_checks: list[dict], chunks: list[dict], message: str) -> list[str]:
    """One fixed sentence per city checked this turn that the message itself names (not one the extractor carried over
    from earlier messages) or that the model checked with the tool, with the KB Section 10 summary's tag when
    present."""
    summary = next((c for c in chunks if c["chunk_type"] == "table_summary"
                    and c["section_no"] == SERVICE_AREA_SECTION), None)
    tag = f" {citation_label(RetrievedChunk(**summary))}" if summary else ""
    lines = []
    for check in [c for c in area_checks if c.get("by") == "tool" or str(c["city"]).lower() in message.lower()]:
        if check["status"] == "listed":
            lines.append(welcome.AREA_LISTED_TEXT.format(city=check.get("matched_city") or check["city"]) + tag)
        else:
            lines.append(welcome.AREA_UNVERIFIED_TEXT.format(city=check["city"]) + tag)
    return lines


def closing_question(state: dict) -> tuple[str, dict]:
    """The one question: the next missing detail; for a customer who only asked questions the fixed offer, once per
    chat and never after a skipped question or a no; otherwise "Anything else you'd like to know?". Returns (question,
    state update)."""
    item = _wanted_item(state)
    if item and item in welcome.NEXT_QUESTION:
        return welcome.NEXT_QUESTION[item], {}
    if (not state.get("lead_id") and not state.get("support_case") and not state.get("fallback_offered")
            and not state.get("ever_declined") and not state.get("skipped_asks")):
        return welcome.FALLBACK_OFFER, {"fallback_offered": True}
    return welcome.ANYTHING_ELSE_ASK, {}


def fallback_reply(state: dict, chunks: list[dict], outcome: str | None) -> tuple[str, dict, list[int]]:
    """The safety-net reply for a turn no model answered: (reply, state update, the KB sections quoted). `chunks` are
    this turn's chunks (city summary first, then retrieval and tool searches); `outcome` is what this turn saved, in
    fixed words with its own question, or None."""
    message = str(state["messages"][-1].content)
    size_match = SIZE_RE.search(message)
    size = re.match(r"\d+(?:\.\d+)?", size_match.group(0)).group(0) if size_match else None
    if not size and state.get("lead", {}).get("system_size_kw"):
        size = f"{state['lead']['system_size_kw']:g}"
    question = " ".join(filter(None, [message, state.get("search_query")]))
    statements = city_lines(state.get("area_checks", []), chunks, message)
    quotable = [c for c in chunks if not (statements and c["section_no"] == SERVICE_AREA_SECTION)]  # said already
    # KB sentences only for a message that asks something (or states a size): a details-only message gets none
    offer = asks_what_is_offered(message)
    picked = kb_sentences(quotable, question, size, offer) if state.get("search_query") or size_match or offer else []
    if picked:
        statements.append(_tagged(picked))
    update: dict = {}
    if outcome:
        statements.append(outcome)
    else:
        if not statements:
            if state.get("new_details"):
                statements.append(welcome.DETAILS_THANKS_TEXT)
            elif state.get("search_query"):
                statements.append(welcome.NOT_COVERED_TEXT)
            else:
                statements.append(welcome.CONTINUE_TEXT)
        ask, update = closing_question(state)
        statements.append(ask)
    sections = list(dict.fromkeys(chunk["section_no"] for _, chunk in picked))
    return "\n\n".join(statements), update, sections

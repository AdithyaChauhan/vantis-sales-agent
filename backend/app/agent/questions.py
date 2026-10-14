"""What a reply may ask or offer (T13e, from Adithya's live test of 562f27e).

Every check here reads our own replies (this one and the earlier ones), the understand step's fields and KB data,
never the customer's words: the AI understands, the code enforces.
- No question that nearly repeats the previous reply's question ("Which city is your property located in?" closed
  four replies in a row).
- Each kind of offer at most once per chat ("Would you like to schedule a site assessment…?" four times), and no
  offer while asks are parked (two skipped questions or offers in a row, or a clear no).
- An explorer is never asked for a callback time or offered a callback; an info-only customer is never asked for name,
  phone or email (KB Section 17).
- One detail per question; only name and phone may go together (KB Section 14).
- A region is never confirmed as a place Vantis covers: its listed cities, then which city (KB Section 10).
last_question_kind() is the backup reading of our last question, for a turn whose understand step failed.
"""
import re

from langchain_core.messages import AIMessage, BaseMessage
from rapidfuzz import fuzz

from app.agent import welcome
from app.agent.leads import (
    CONTACT_PAIR, _ITEM_WORDS, _wanted_item, ask_targets, ask_text, asked_items, asks_parked, is_explorer,
    ordered_missing, region_turn, shows_interest,
)
from app.rag.answer import citation_label, split_sentences, without_tags
from app.rag.chunker import SERVICE_AREA_SECTION
from app.rag.retriever import RetrievedChunk
from app.services.service_areas import region_patterns

# A question this close to one of the previous reply's questions repeats it (fuzz.ratio on lowercase letters and
# digits). Measured: word for word 100, "your property" vs "the property" 91; "name and phone number?" then "phone
# number?" 88 is a real follow-up and stays
NEAR_REPEAT_SCORE = 90
# An offer: a question in our reply that proposes a next step ...
_OFFER_RE = re.compile(
    r"^(?:so,?\s+|and\s+|also,?\s+)?(?:would\s+you\s+like|would\s+you\s+be\s+interested|are\s+you\s+interested|"
    r"do\s+you\s+want|shall\s+(?:i|we)|should\s+i|can\s+i\s+help|may\s+i\s+help|want\s+me\s+to|would\s+it\s+help|"
    r"how\s+about|interested\s+in)\b", re.I)
# ... unless it asks the customer to share a detail ("Would you like to share your bill?" asks for the bill)
_SHARE_RE = re.compile(r"\b(?:share|tell\s+me|give\s+me|send|provide)\b[^?]*\b(?:your|the)\b", re.I)
OFFER_KINDS = (  # (kind, its words), checked in this order: the first that matches is the offer's kind
    ("support request", re.compile(r"\bsupport\s+request\b", re.I)),
    ("site assessment", re.compile(r"\bsite\s+(?:assessment|visit|survey|inspection)\b|\bassessment\b", re.I)),
    ("quote", re.compile(r"\bquot(?:e|es|ation)\b|\bestimate\b", re.I)),
    ("callback", re.compile(r"\bcall\s*-?\s*backs?\b|\b(?:a|the)\s+call\b|\bcall\s+you\b|\bring\s+you\b", re.I)),
    ("city check", re.compile(r"\b(?:check|confirm|verify)\b[^?]*\b(?:city|cities|area|location|town)\b", re.I)),
    ("packages", re.compile(r"\bpackages?\b|\breference\s+options?\b", re.I)),
    ("sales team", re.compile(r"\bteam\b|\bconnect\s+you\b|\bput\s+you\s+in\s+touch\b|\bexpert\b|\brepresentative\b",
                              re.I)),
)
# A question that leaves the next step to the customer ("Anything else I can help with?")
_OPEN_RE = re.compile(r"\b(?:anything|something)\s+else\b|\bany\s+(?:other|more)\s+questions?\b|\bkuch\s+aur\b", re.I)
_YES_NO_RE = re.compile(r"^(?:is|are|am|was|were|do|does|did|can|could|would|will|shall|should|may|might|have|has|"
                        r"kya)\b", re.I)
# A question about a callback, or its time (an explorer is asked neither)
_CALLBACK_ASK_RE = re.compile(r"\bcall\s*-?\s*backs?\b|\bcall\s+you\b|\b(?:what|which)\s+(?:time|day)\b|"
                              r"\bwhen\b[^?]*\bcall\b", re.I)
_READY_RE = re.compile(r"\bwhen(?:ever)?\s+you(?:['’]re|\s+are)\s+ready\b", re.I)
# The details a question asks the customer to share, by their nouns (to count them in one question)
_ASKED_DETAILS = {
    "city": re.compile(r"\b(?:city|town)\b", re.I),
    "name": re.compile(r"\bname\b", re.I),
    "phone or email": re.compile(r"\b(?:phone|mobile|number|e-?mail)\b", re.I),
    "monthly bill": re.compile(r"\b(?:bill|usage|consumption|units)\b", re.I),
    "timeline": re.compile(r"\btimeline\b", re.I),
}
_CONTACT_DETAILS = ("name", "phone or email")
_CITY_QUESTION_RE = re.compile(r"\b(?:city|town)\b", re.I)
# An opening that answers "do you cover my area?" with a yes (dropped on a region turn)
_YES_RE = re.compile(r"^\W*(?:yes|yeah|yep|yup)\b", re.I)
_YOUR_AREA_RE = re.compile(r"\byour\s+(?:area|region|state)\b", re.I)


def question_sentences(text: str) -> list[str]:
    """The question sentences of a text, without their KB tags."""
    pieces = [without_tags(piece) for line in text.split("\n") for piece in split_sentences(line)]
    return [piece for piece in pieces if piece.endswith("?")]


def previous_reply(messages: list[BaseMessage]) -> AIMessage | None:
    """The assistant's last reply before the customer's new message."""
    return next((m for m in reversed(messages[:-1]) if isinstance(m, AIMessage)), None)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def repeats(question: str, earlier: list[str]) -> bool:
    """True when `question` is word for word, or nearly, one of the `earlier` questions (NEAR_REPEAT_SCORE)."""
    return any(fuzz.ratio(_norm(question), _norm(q)) >= NEAR_REPEAT_SCORE for q in earlier)


def offer_kind(question: str) -> str | None:
    """The kind of next step a question of ours offers (OFFER_KINDS; "other" when none fits), or None when it is not
    an offer."""
    text = without_tags(question).strip()
    if not _OFFER_RE.search(text) or _SHARE_RE.search(text):
        return None
    return next((kind for kind, pattern in OFFER_KINDS if pattern.search(text)), "other")


def offers_made(messages: list[BaseMessage]) -> list[str]:
    """The kinds of offer our earlier replies in this chat made, in order (never "other")."""
    kinds: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for question in question_sentences(str(message.content)):
                kind = offer_kind(question)
                if kind and kind != "other" and kind not in kinds:
                    kinds.append(kind)
    return kinds


def made_offer(previous: object) -> bool:
    """True when our last reply offered a next step."""
    return isinstance(previous, AIMessage) and any(offer_kind(q) for q in question_sentences(str(previous.content)))


def asked_contact(previous: object) -> bool:
    """True when our last reply asked for the customer's name, phone or email."""
    if not isinstance(previous, AIMessage):
        return False
    return any(_ASKED_DETAILS[item].search(q) for q in question_sentences(str(previous.content))
               for item in _CONTACT_DETAILS)


def previous_asked(state: dict, previous: object) -> bool:
    """True when our last reply asked for a detail the lead still needed or made an offer (T13d, T13e): a reply that
    does neither leaves the skip count as it is."""
    if not isinstance(previous, AIMessage):
        return False
    return bool(asked_items(str(previous.content), ask_targets(state))) or made_offer(previous)


def last_question_kind(previous: object) -> str | None:
    """Backup only (T13e: the understand step's reply_type decides; this runs when that step failed): what our last
    reply's final question was: "open" ("Anything else I can help with?"), "offer", "detail" (asks for one of their
    details), "yes_no", or None when it asked nothing. Reads our own reply, never the customer's words."""
    questions = question_sentences(str(previous.content)) if isinstance(previous, AIMessage) else []
    if not questions:
        return None
    question = questions[-1]
    if _OPEN_RE.search(question):
        return "open"
    if offer_kind(question):
        return "offer"
    if asked_items(question, list(_ITEM_WORDS)):
        return "detail"
    return "yes_no" if _YES_NO_RE.search(question) else "open"


def _details_asked(question: str) -> list[str]:
    return [item for item, pattern in _ASKED_DETAILS.items() if pattern.search(question)]


def _one_detail_question(state: dict) -> str | None:
    """The fixed question for the next missing detail, or name and phone together when those two are next."""
    lead = state.get("lead", {})
    if not state.get("lead_id") and not state.get("support_case"):
        if ordered_missing(lead, state.get("contact_request", False))[:2] == CONTACT_PAIR:
            return welcome.CONTACT_PAIR_ASK
    item = _wanted_item(state)
    return ask_text(item, state) if item else None


def _why_not(question: str, state: dict, earlier: list[str], made: list[str]) -> str | None:
    """Why a question of this reply must go, or None when it may stay."""
    if not state.get("validation_errors") and repeats(question, earlier):
        return "repeats the last reply's question"
    kind = offer_kind(question)
    if is_explorer(state) and (kind == "callback" or _CALLBACK_ASK_RE.search(question)):
        return "an explorer is never asked about a callback"
    if kind and asks_parked(state):
        return "no offers after skipped or declined ones"
    if kind and kind != "other" and kind in made:
        return f"the {kind} was offered before"
    if kind:
        return None
    asked = _details_asked(question)
    info_only = (not state.get("lead_id") and not state.get("support_case") and not shows_interest(state)
                 and not state.get("validation_errors") and "gives_details" not in state.get("intents", []))
    if info_only and any(item in _CONTACT_DETAILS for item in asked):
        return "an info-only customer is not asked for contact details"
    if len(asked) > 1 and sorted(asked) != sorted(_CONTACT_DETAILS):
        return "more than one detail in one question"
    return None


def ask_rules(reply: str, state: dict) -> tuple[str, list[dict]]:
    """Apply the question and offer rules to a reply (T13e, see the module docstring). A question asking for more than
    one detail is replaced by the one-detail question; every other failing question is removed, with its tags.
    Returns (reply, [{"text", "why"}] for each question changed); never an empty reply."""
    messages = state.get("messages", [])
    previous = previous_reply(messages)
    earlier = question_sentences(str(previous.content)) if previous is not None else []
    made = offers_made(messages)
    lines, changed, explorer_cut = [], [], False
    for line in reply.split("\n"):
        kept, dropped_last = [], False
        for piece in split_sentences(line):
            text = without_tags(piece)
            if not text:  # tags alone follow the sentence before them
                if piece.strip() and not dropped_last:
                    kept.append(piece.strip())
                continue
            why = _why_not(text, state, earlier, made) if text.endswith("?") else None
            dropped_last = why is not None
            if why is None:
                kept.append(piece.strip())
                continue
            changed.append({"text": text, "why": why})
            explorer_cut = explorer_cut or why.startswith("an explorer")
            if why == "more than one detail in one question":
                replacement = _one_detail_question(state)
                if replacement and not repeats(replacement, earlier):
                    kept.append(replacement)
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    if not changed:
        return reply, []
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if (explorer_cut or not text) and not _READY_RE.search(text):
        text = f"{text}\n\n{welcome.EXPLORER_READY_TEXT}".strip()
    return text, changed


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def region_line(check: dict, tag: str = "") -> str:
    """The fixed line for a region check: its listed cities (KB Section 10 data), with `tag` before the full stop."""
    region = (check.get("regions") or [check["city"]])[0]
    line = welcome.AREA_REGION_TEXT.format(region=region, cities=_join(check.get("region_cities") or [region]))
    return f"{line[:-1]} {tag}." if tag else line


def fix_region_reply(reply: str, state: dict, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
    """A region is never a place Vantis covers (T13e, KB Section 10). On a turn that named one with no city known yet,
    every statement of ours that names the region (its KB name, a short form, the customer's spelling, or "your area")
    is replaced, once, by the fixed line with the region's listed cities, which says it correctly, so a replaced
    sentence costs nothing; an opening "Yes…" is dropped (it answered "do you cover my area?"); the line is added when
    the reply doesn't name all the cities; and the reply asks which city (unless asks are parked). Reads only our own
    reply. Returns (reply, what changed)."""
    if not region_turn(state) or state.get("support_case"):
        return reply, []
    checks = [c for c in state.get("area_checks", []) if c.get("status") == "region"]
    summary = next((c for c in chunks if c.section_no == SERVICE_AREA_SECTION and c.chunk_type == "table_summary"),
                   next((c for c in chunks if c.section_no == SERVICE_AREA_SECTION), None))
    lines_text = [region_line(c, citation_label(summary) if summary else "") for c in checks]
    patterns = [p for c in checks for region in (c.get("regions") or []) for p in region_patterns(region)]
    patterns += [re.compile(rf"\b{re.escape(c['city'])}\b", re.I) for c in checks
                 if c["city"].replace(".", "").replace(" ", "").lower() != "up"]  # the customer's own spelling
    patterns.append(_YOUR_AREA_RE)
    cities = [city for c in checks for city in c.get("region_cities", [])]

    def names_region(text: str) -> bool:
        return any(p.search(text) for p in patterns)

    def named_cities(text: str) -> list[str]:  # the region's own name ("Delhi" in "Delhi NCR") is not one of its cities
        for pattern in patterns:
            text = pattern.sub(" ", text)
        return [city for city in cities if re.search(rf"\b{re.escape(city)}\b", text, re.I)]

    changed: list[str] = []
    lines, placed, opening = [], False, True
    for line in reply.split("\n"):
        kept, dropped_last = [], False
        for piece in split_sentences(line):
            text = without_tags(piece)
            if not text:  # tags alone go with the sentence before them
                if piece.strip() and not dropped_last:
                    kept.append(piece.strip())
                continue
            first, opening = opening, False
            dropped_last = not text.endswith("?") and (names_region(text) or (first and bool(_YES_RE.match(text))))
            if not dropped_last:
                kept.append(piece.strip())
                continue
            changed.append(text)
            if names_region(text) and not placed:
                kept.extend(lines_text)
                placed = True
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if not placed and len(set(named_cities(text))) < len(set(cities)):  # every listed city of the region, or the line
        text = "\n\n".join(lines_text + ([text] if text else []))
        changed.append("added the region's cities")
    if not asks_parked(state) and not any(_CITY_QUESTION_RE.search(q) for q in question_sentences(text)):
        region = (checks[0].get("regions") or [checks[0]["city"]])[0]
        text = f"{text} {welcome.REGION_CITY_ASK.format(region=region)}".strip()
        changed.append("asked which city")
    return text, changed

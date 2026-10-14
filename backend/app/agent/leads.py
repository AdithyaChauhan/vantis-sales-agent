"""Lead state for the agent: the status block the model reads, city confirmation, short answers, the callback
time, the write-once sync rule, the guard against false "saved" claims and the lead summary for the API.
Calls the T7 leads service; never decides scores itself.

The model never sees a score, a status, a reason or any personal value: the status block lists field
names and plain messages only.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from sqlalchemy import Engine
from sqlalchemy.exc import DBAPIError

from app.agent import welcome
from app.leads import service
from app.leads.schemas import LeadData
from app.leads.scoring import missing_details, missing_for_creation, score_lead
from app.rag.answer import is_claim, split_sentences, without_tags
from app.services.service_areas import ServiceAreas

logger = logging.getLogger(__name__)

# The order F2 asks for what is missing; a contact request puts name and phone first
NEED_ORDER = ["city", "requirement, monthly bill, usage, system size or property type", "name", "phone or email"]
CONTACT_ORDER = ["name", "phone or email", "city", "requirement, monthly bill, usage, system size or property type"]
SUPPORT_NEEDS = ("name", "phone or email", "issue", "project details")
BUYING_INTENTS = {"buying_interest", "contact_request", "asks_what_is_needed"}
DETAIL_INTENTS = {"buying_interest", "contact_request", "gives_details"}  # interest shown after the lead is saved
INFO_ONLY_LINE = ("Lead saved: no. The customer has only asked for information so far: do not ask for their "
                  "details, and never say they are saved; after answering, offer one next step related to what "
                  "they asked.")
SAVED_NOW_LINE = ("Next: say the details are saved and the sales team can follow up, then offer a callback (ask "
                  "what time suits them). Do not ask for more details in this reply.")
SAVED_WITH_TIME_LINE = ("Next: call schedule_callback for the time they gave, then say the details are saved and the "
                        "callback request is logged; the team will confirm it. Do not ask for more details.")
CALLBACK_NOW_LINE = ("Next: confirm what the tools did in this turn (the callback request logged or updated for "
                     "their time, the team will confirm it; and the details saved, if create_lead ran). Do not ask "
                     "for more details in this reply.")
ASKED_WHAT_LINE = ("The customer asked what else you need: if they asked about one of their details, say whether it "
                   "would still help and ask for it; otherwise ask now for {detail}, in everyday words. One question.")
# T13e: a customer only exploring or researching is never asked for a callback time (KB Section 17)
SAVED_EXPLORING_LINE = ("Next: say the details are saved and that the Vantis team can help whenever they are ready. Do "
                        "not ask for a callback time or for more details, and offer nothing.")
# T13e: the understand step's problem_now is "unsure": no support flow yet, the general answer and one question
PROBLEM_UNCLEAR_LINE = ("The customer mentioned their installed system, but it isn't clear whether something is wrong "
                        "with it now: give the general support answer from the context, then ask whether something is "
                        "wrong with it right now. Do not ask for their details yet.")
INTEREST_LINE = "Next: after answering anything they asked, ask gently for {detail}. One question."
NOTHING_LEFT_LINE = ("The customer asked what else you need: nothing else is needed for now (they ruled out the rest); "
                     "say so and offer one next step.")
SAVED_INFO_LINE = ("The customer only asked for information or said no: answer and offer one next step related to "
                   "it; do not ask for these details.")
NOTHING_MORE_LINE = "Nothing more is needed for the lead: answer and offer one next step."
# T13d: after one skipped question about their details the assistant may ask once more; after two it stops asking
# until they answer, ask to be contacted, ask what is needed or want to go ahead (Adithya's live chat: the city was
# asked 4 times while the customer kept asking other things). T13e: offers count too, and a clear no parks at once
MAX_SKIPPED_ASKS = 2
SKIPPED_LINE = ("The customer declined or skipped your recent questions or offers: answer what they ask, with no "
                "question about their details and no offer.")
# T13e: name and phone may be asked in one question; every other question asks for one detail (KB Section 14)
CONTACT_PAIR = ["name", "phone or email"]
# T13a: a city the customer only asked about counts as theirs
ASKED_CITY_LINE = "City: the one they asked about counts as theirs; never ask for it or confirm it again."
MAX_CITY_CANDIDATES = 3
# T13b: the property type's word in the package rule (KB Section 2: homes are residential, shops and offices commercial)
PROPERTY_OF = {"residential": "home", "commercial": "business"}
# The details that make a turn a package turn: the bill, or the property type given after it
PACKAGE_DETAILS = {"monthly_bill_inr", "customer_type", "property_type"}
# A question that asks for the city, or one that confirms a known city (drop_known_city_question, T13a)
_CITY_ASK_RE = re.compile(
    r"\b(?:which|what)\s+(?:city|town|location|area)\b|\byour\s+(?:city|town|location)\b|\blocated\b|"
    r"\bwhere\s+(?:is|are|do|will|would)\b[^?]*\b(?:home|house|property|based|live|living|shop|office|business|"
    r"installation)\b", re.I)


def _confirms_city_re(city: str) -> re.Pattern:
    """'Is the property in Lucknow?', 'your home in Lucknow', 'Lucknow, right?'"""
    name = re.escape(city)
    return re.compile(rf"\b(?:is|are)\s+(?:it|this|that|the\s+\w+|your\s+\w+)\s+in\s+{name}\b|"
                      rf"\byour\s+(?:home|house|property|place|shop|office|business|building)\s+in\s+{name}\b|"
                      rf"\b{name}\s*,?\s*(?:right|correct)\b", re.I)
_QUESTION_RE = re.compile(r"[^.!?]*\?")
# Short answers (T12a): a bare amount, or a bare yes or no
_BARE_AMOUNT_RE = re.compile(r"^\s*(?:₹|rs\.?|inr)?\s*\d[\d,]*(?:\.\d+)?\s*(?:k|rs\.?|rupees|/-)?[\s.!]*$", re.I)
_YES_RE = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|of course|it does|we do|yes,? it does|yes,? we do)[\s.!]*$", re.I)
_NO_RE = re.compile(r"^\s*(?:no|nope|nah|naah|it doesn't|we don't|no,? it doesn't|no,? we don't)[\s.!]*$", re.I)
# T13c: the kind of place, only with confidence (place_type). Adithya's word lists; "ghar" is Hindi for home; BHK
# sizes, bungalow and duplex added in planning. "residential" and "commercial" count only as a short answer. T13d: the
# KB's other small-business places (clinics, warehouses) and Hindi words: makan/kothi (house), dukan (shop), daftar
# (office), vyapar (business), godown (warehouse); "meri dukan" counted as no kind of place in Adithya's live chat
_HOME = r"home|house|flat|apartment|villa|ghar|makaan|makan|kothi|bungalow|duplex|\d+\s*-?\s*bhk"
_BUSINESS = (r"shop|office|store|business|clinic|warehouse|godown|showroom|factory|dukan|dukaan|dukkan|daftar|"
             r"vyapar|vyapaar")
_PLACE = rf"(?:{_HOME}|{_BUSINESS})"
_HOME_RE = re.compile(rf"\b(?:{_HOME}|residential)\b", re.I)
_BUSINESS_RE = re.compile(rf"\b(?:{_BUSINESS}|commercial)\b", re.I)
_APOS = "['’]"
# Passing mentions, never their own place: being somewhere ("I'm at home right now", "I'm at my office"), "at home" or
# "from home", working from or at a place, a home or business loan, someone else's place ("my friend's house", "dost
# ke ghar")
_PASSING_RE = re.compile(
    rf"\b(?:i{_APOS}?m|i\s+am|we{_APOS}?re|we\s+are)\s+(?:at|in)\s+(?:my\s+|our\s+|the\s+)?{_PLACE}\b|"
    rf"\b(?:at|from)\s+home\b|"
    rf"\bwork(?:s|ing)?\s+(?:from|at|in)\s+(?:my\s+|our\s+|the\s+|an?\s+)?{_PLACE}\b|"
    rf"\b{_PLACE}\s+loans?\b|"
    rf"\b(?!(?:it|that|this|what|there|here|who|he|she|let){_APOS}s\b)[a-z]+{_APOS}s\s+(?:own\s+)?{_PLACE}\b|"
    rf"\b[a-z]+\s+(?:ka|ke|ki)\s+ghar\b", re.I)
# Their own place: "my flat", "mere ghar", "it's for home", "the house is in Kanpur", "we live in a flat", "I have a
# 3BHK", "it's a shop" (the place word ends the phrase, so "that's a business decision" is not one)
_OWN_PLACE_RE = re.compile(
    rf"\b(?:my|our|mera|mere|meri|hamara|hamare|apna|apne)\s+(?:own\s+|new\s+|small\s+)?(?P<a>{_PLACE})\b|"
    rf"\bfor\s+(?:the\s+)?(?P<b>home)\b|"
    rf"\b(?:the|this)\s+(?P<c>{_PLACE})\s+is\s+(?:in|at|near|on|located)\b|"
    rf"\b(?:i|we)\s+(?:live|stay|reside)\s+in\s+(?:an?\s+|my\s+|our\s+|the\s+)?(?P<d>{_PLACE})\b|"
    rf"\b(?:i|we)\s+(?:have|own|run)\s+(?:an?\s+)(?:small\s+|big\s+|new\s+)?(?P<e>{_PLACE})\b|"
    rf"\b(?:it|this|that)(?:{_APOS}s|\s+is)\s+(?:an?|my|our)\s+(?:small\s+|big\s+|new\s+)?(?P<f>{_PLACE})"
    rf"(?=\s*(?:$|[.,!?;)]|\s+(?:in|at|near|on|with|and|so|but)\b))", re.I)
# A Hindi place word counts only in a message about solar or installing ("solar lagwana hai ghar par", T13d: "dukan par
# solar lagwana hai"); "main ghar par hoon" is only "I'm at home"
_GHAR_RE = re.compile(r"\b(?:ghar|makaan|makan|kothi|dukan|dukaan|dukkan|daftar)\b", re.I)
_SOLAR_WORDS_RE = re.compile(r"\b(?:solar|panels?|lagwa\w*|laga\w*|install\w*)\b", re.I)
SHORT_ANSWER_WORDS = 6  # "a small shop in Lucknow" is still an answer to the question
# The detail a question asks about, by its words (only a question about one detail is read)
_TOPIC_OF = {
    "bill": "bill", "roof": "roof", "rooftop": "roof", "kw": "size", "size": "size", "units": "usage",
    "kwh": "usage", "usage": "usage", "consumption": "usage", "name": "name", "phone": "phone", "number": "phone",
    "email": "email", "city": "city", "when": "timeline", "timeline": "timeline", "own": "ownership",
    "rent": "ownership", "renting": "ownership",
}
_TOPIC_RE = re.compile(r"\b(" + "|".join(_TOPIC_OF) + r")\b", re.I)
# Callback times (T12a): a day, and an hour or part of the day
_DAY_RE = re.compile(r"\b(?:day after tomorrow|today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|"
                     r"saturday|sunday|this week|next week|weekend|\d{1,2}(?:st|nd|rd|th))\b", re.I)
_CLOCK_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)|\b\d{1,2}:\d{2}\b|\b(?:noon|midday|"
                       r"morning|afternoon|evening|night)\b", re.I)
# A time of day alone ("4pm", "around 6 pm"): a time to call, never an installation timeline (T12a replay)
_CLOCK_ONLY_RE = re.compile(r"^\s*(?:at|around|about|by|after|before)?\s*(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|"
                            r"p\.m\.)|\d{1,2}:\d{2}|noon|midday|morning|afternoon|evening|night)[\s.!]*$", re.I)
# What a reply may say about a callback (T12a): only what schedule_callback did
_CALLBACK_WORD_RE = re.compile(r"\bcall[\s-]?backs?\b|\bcall request\b", re.I)
_CALLBACK_DONE_RE = re.compile(r"\b(?:logged|booked|scheduled|arranged|set up|updated|confirmed|registered|saved|"
                               r"noted)\b", re.I)
_CALLBACK_FIRM_RE = re.compile(r"\b(?:booked|scheduled|arranged|set up|confirmed|updated)\b", re.I)
# Details the customer names or rules out ("apart from timing"), by their words (T12a)
_EXCEPT_RE = re.compile(r"\b(?:apart from|except(?: for)?|other than|besides|aside from|excluding)\s+"
                        r"(?:the\s+|my\s+|your\s+)?([a-z]+)", re.I)
_DETAIL_WORDS = {
    "timeline": ("timing", "timeline", "time", "when", "date"),
    "monthly bill or usage": ("bill", "usage", "units", "consumption"),
    "rooftop availability": ("roof", "rooftop", "terrace"),
    "ownership": ("ownership", "own", "owner", "rent", "renting", "rented"),
    "email": ("email", "mail"),
    "phone number": ("phone", "mobile", "number"),
}
_DETAIL_OF_WORD = {word: detail for detail, words in _DETAIL_WORDS.items() for word in words}
# The words of each still-missing item, to find the question that asks for it (keep_one_question, T12b)
_ITEM_WORDS = {
    "city": ("city", "where", "location", "town"),
    "requirement, monthly bill, usage, system size or property type": (
        "bill", "usage", "units", "consumption", "size", "kw", "property", "house", "home", "shop", "office"),
    "name": ("name",),
    "phone or email": ("phone", "number", "mobile", "email", "contact"),
    "issue": ("issue", "problem", "happening", "wrong"),
    "project details": ("system", "installed", "size", "details"),
    "property type": ("home", "house", "business", "property", "shop", "office", "residential", "commercial"),
    **_DETAIL_WORDS,
}


@dataclass
class LeadOps:
    """What the lead tools and the finalize sync use: the leads database and the reason model.

    Live tests pass the Neon test branch here, so they never write to the main database.
    """

    engine: Engine
    areas_fn: Callable[[], ServiceAreas]
    reason_llm: Runnable | None = None  # None: the service uses reason_llm() (the main model, 4 s)
    reason_now: bool = True  # False: save the template reason, the API refreshes it after replying
    calls: list[str] = field(default_factory=list)  # every write, for tests ("create", "update", ...)


def ordered_missing(lead: dict, contact_request: bool) -> list[str]:
    """What is still needed before the lead can be saved, in the order to ask for it."""
    missing = set(missing_for_creation(LeadData.model_validate(lead)))
    order = CONTACT_ORDER if contact_request else NEED_ORDER
    return [item for item in order if item in missing]


def support_missing(lead: dict, support: dict) -> list[str]:
    """What an existing customer's support request still needs (KB Sections 11 and 12)."""
    have = {
        "name": bool(lead.get("name")),
        "phone or email": bool(lead.get("phone") or lead.get("email")),
        "issue": bool(support.get("issue")),
        "project details": bool(support.get("project_details")),
    }
    return [need for need in SUPPORT_NEEDS if not have[need]]


def asks_parked(state: dict) -> bool:
    """True once the customer skipped two questions about their details in a row (T13d): no more asks for now."""
    return state.get("skipped_asks", 0) >= MAX_SKIPPED_ASKS


def ask_targets(state: dict) -> list[str]:
    """Everything the assistant may still ask the customer for: the support request's needs, or the lead's required
    details before the save, or after the save the details that would still add points. Item names are _ITEM_WORDS
    keys. (The bill turn's "home or business?" is not here: that turn gave a detail, so nothing was skipped.)"""
    lead = state.get("lead", {})
    if state.get("support_case") and not state.get("support_request_id"):
        return support_missing(lead, state.get("support", {}))
    if not state.get("lead_id"):
        return ordered_missing(lead, state.get("contact_request", False))
    ruled_out = set(state.get("ruled_out", []))
    return [d for d in missing_details(LeadData.model_validate(lead), _breakdown_stub(lead)) if d not in ruled_out]


def asked_items(text: str, items: list[str]) -> list[str]:
    """The items a question sentence of `text` asks for, by their words (_ITEM_WORDS), in the order of `items`."""
    questions = [without_tags(piece) for line in text.split("\n") for piece in split_sentences(line)]
    questions = [q for q in questions if q.endswith("?")]
    return [item for item in items
            if any(re.search(rf"\b{w}\b", q, re.I) for q in questions for w in _ITEM_WORDS.get(item, ()))]


def skipped_asks(state: dict, previous_asked: bool, reply_type: str, reset: bool, parks: bool) -> int:
    """The count of questions and offers the customer skipped or declined in a row, after this message (T13d). T13e:
    read from the understand step's reply_type and our own last reply, never from the customer's words:
    - parks: a clear no to an offer or to sharing details: MAX_SKIPPED_ASKS at once (no asks, no offers);
    - reset: an answer, a yes, going ahead, a contact request, "what else do you need", a new callback time: 0. A
      "declines" that doesn't park (a no that answered a question, or one detail refused) counts as an answer;
    - otherwise (skips, asks something new, closes) +1 when our last reply asked for a detail or made an offer."""
    if parks:
        return MAX_SKIPPED_ASKS
    if reset or reply_type == "declines":
        return 0
    return state.get("skipped_asks", 0) + (1 if previous_asked else 0)


def shows_interest(state: dict) -> bool:
    """The customer wants solar for their own place: a buying or contact intent (or asking what else is needed)
    this turn, or any detail about themselves in the draft (intent_level alone is the extractor's guess). T13e: a city
    they only asked about ("Do you provide installation in <city>?") is an information question, not interest."""
    if set(state.get("intents", [])) & BUYING_INTENTS:
        return True
    asked_city = state.get("city_source") == "asked"
    return any(name != "intent_level" and not (name == "city" and asked_city) for name in state.get("lead", {}))


def is_explorer(state: dict) -> bool:
    """T13e: the customer said they are only exploring or researching (the understand step's intent_level) and asked for
    no call: they are never asked for a callback time; the team can help whenever they are ready."""
    return (state.get("lead", {}).get("intent_level") == "exploring" and not state.get("call_requested")
            and not state.get("callback_time"))


def region_turn(state: dict) -> bool:
    """T13e: this turn named a KB region ("Delhi NCR") and the city is still unknown: the next ask is which city."""
    return not state.get("lead", {}).get("city") and any(
        c.get("status") == "region" for c in state.get("area_checks", []))


def ask_text(item: str, state: dict) -> str | None:
    """The fixed one-detail question for a missing item (T13e: a ruled-out phone asks for the email, and back)."""
    ruled_out = set(state.get("ruled_out", []))
    if item == "phone or email" and "phone number" in ruled_out:
        return welcome.NEXT_QUESTION["email"]
    if item == "phone or email" and "email" in ruled_out:
        return welcome.NEXT_QUESTION["phone number"]
    return welcome.NEXT_QUESTION.get(item)


def ask_items(missing: list[str]) -> list[str]:
    """What one question may ask for (T13e, KB Section 14): the first missing item, or name and phone together."""
    return missing[:2] if missing[:2] == CONTACT_PAIR else missing[:1]


def pending_changes(lead: dict, saved: dict) -> dict:
    """Draft fields that differ from what was last written to the database."""
    return {k: v for k, v in lead.items() if saved.get(k) != v}


def changed_fields(before: dict, after: dict) -> list[str]:
    """The lead fields this turn stated or changed (intent_level, the extractor's guess, does not count)."""
    return sorted(k for k, v in after.items() if before.get(k) != v and k != "intent_level")


def final_question(message) -> str:
    """The last question sentence of an assistant message ('' if it asked none)."""
    if not isinstance(message, AIMessage):
        return ""
    questions = _QUESTION_RE.findall(str(message.content))
    return questions[-1] if questions else ""


def confirmed_city(asked_cities: list[str], previous: object, out: dict) -> str | None:
    """A candidate city the customer confirmed: the assistant's last question named exactly one of the candidates
    (T13a: "yes" to "Lucknow or Kanpur?" picks nothing), and the reply gives no other city, isn't off-topic and
    doesn't say no. None otherwise."""
    if not asked_cities or out["details"].get("city") or out.get("declines"):
        return None
    if "off_topic" in out["intents"]:
        return None
    question = final_question(previous).lower()
    named = [city for city in asked_cities if city.lower() in question]
    return named[0] if len(named) == 1 else None


def _distinct(names: list[str], key: Callable[[str], str]) -> list[str]:
    """The names without repeats (by `key`), in order, keeping the first spelling."""
    seen, kept = set(), []
    for name in names:
        if key(name) not in seen:
            seen.add(key(name))
            kept.append(name)
    return kept


def resolve_city(
    *, current: str | None, source: str | None, candidates: list[str], stated: str | None, asked: list[str],
    saved: bool, key: Callable[[str], str] = str.lower,
) -> tuple[str | None, str | None, list[str]]:
    """The customer's city after this message (T13a): (city, city_source, candidates).

    - A stated city ("I'm in Kanpur", or a candidate they confirmed) is theirs ("stated"). The same city as an asked
      one stays "asked" (the extractor often repeats the city from the history).
    - Before the lead is saved, a city they only asked about counts as theirs ("asked"), unless more than one city
      is in question (this message's and the earlier asked one together): then none is guessed, and they become
      the candidates, so the bot asks which one when the city is the next missing detail.
    - A question never replaces a stated city, and after the save it changes nothing.
    `key` decides when two names are the same city (the graph uses the service-area match: Gurgaon is Gurugram).
    """
    if stated and not (source == "asked" and current and key(stated) == key(current)):
        return stated, "stated", []
    if not asked or source == "stated" or saved:
        return current, source, candidates
    pool = _distinct(([current] if current else []) + candidates + asked, key)
    if len(pool) == 1:
        return pool[0], "asked", []
    return None, None, pool[-MAX_CITY_CANDIDATES:]


def short_answer_details(previous: object, message: str, details: dict) -> dict:
    """Backup only (T13e: the understand step's short-answer rule decides; this runs when that step failed): a detail
    the customer gave as a bare answer to the assistant's last question (T12a): an amount after a question only about
    the monthly bill, yes or no after a question only about roof space. Values are validated later as usual."""
    topics = {_TOPIC_OF[word.lower()] for word in _TOPIC_RE.findall(final_question(previous))}
    if topics == {"bill"} and "monthly_bill_inr" not in details and _BARE_AMOUNT_RE.match(message):
        return {"monthly_bill_inr": message.strip().rstrip(".!")}
    if topics == {"roof"} and "rooftop_available" not in details:
        if _YES_RE.match(message):
            return {"rooftop_available": True}
        if _NO_RE.match(message):
            return {"rooftop_available": False}
    return {}


def _kinds(words: list[str]) -> set[str]:
    """The customer types the place words name: {"residential"}, {"commercial"}, both or none."""
    return ({"residential"} if any(_HOME_RE.fullmatch(w.strip()) for w in words) else set()) | (
        {"commercial"} if any(_BUSINESS_RE.fullmatch(w.strip()) for w in words) else set())


def passing_place_kinds(message: str) -> set[str]:
    """Backup only (see place_type): the kinds of place a message names only in passing ("I work from home" ->
    {"residential"})."""
    return _kinds([w for m in _PASSING_RE.finditer(message) for w in _HOME_RE.findall(m.group(0))
                   + _BUSINESS_RE.findall(m.group(0))])


def place_type(message: str, previous: object) -> dict:
    """Backup only (T13e: the understand step's place rules decide; this runs when that step failed). The customer type
    from the words of a message, only with confidence (T13c). Home words -> residential, shop, office, store or
    business -> commercial:
    - (a) a short answer to the assistant's own home-or-business question ("house", "a shop", "ghar"), or
    - (b) a phrase about their own place: "the house is in Kanpur", "it's for my home", "my flat", "it's a shop",
      "solar lagwana hai ghar par".
    Passing mentions are removed first and never count ("I'm at home right now", "my friend's house has solar", "I
    work from home", "home loan"). Nothing when both kinds or neither are named: the assistant asks instead."""
    text = _PASSING_RE.sub(" ", message)
    question = final_question(previous)
    asked = bool(_HOME_RE.search(question) and _BUSINESS_RE.search(question))
    if asked and len(re.findall(r"[\w-]+", text)) <= SHORT_ANSWER_WORDS:
        kinds = _kinds(_HOME_RE.findall(text) + _BUSINESS_RE.findall(text))
    else:
        found = [w for m in _OWN_PLACE_RE.finditer(text) for w in m.groups() if w]
        if _SOLAR_WORDS_RE.search(text):
            found += [m.group(0) for m in _GHAR_RE.finditer(text)]
        # the place named in passing elsewhere in the message still counts, so "my home office" names both kinds
        kinds = _kinds(found) | (_kinds(_HOME_RE.findall(text) + _BUSINESS_RE.findall(text)) if found else set())
    return {"customer_type": kinds.pop()} if len(kinds) == 1 else {}


def chat_intent(lead: dict, intents: set[str], callback_time: str | None) -> dict:
    """The intent the chat itself shows, over the extractor's guess:
    - T13d: a request to be called (a contact request, or a callback time) is "ready", whatever the extractor said
      (the assignment's Hot is "strong purchase intent"; audit S21: "can someone call me?" plus details scored COLD 43);
    - T13c: buying interest while the intent is unknown is "interested" (chat 1 scored 48 with the intent unknown after
      the customer asked the price).
    {} when nothing changes."""
    if "contact_request" in intents or callback_time:
        return {} if lead.get("intent_level") == "ready" else {"intent_level": "ready"}
    if lead.get("intent_level") or not intents & BUYING_INTENTS:
        return {}
    return {"intent_level": "interested"}


def merge_callback_time(old: str | None, new: str | None) -> str | None:
    """The customer's preferred callback time after this message (T12a). A new time replaces the old one,
    except that a day alone keeps the earlier hour and an hour alone keeps the earlier day:
    "4pm" then "day after tomorrow" gives "day after tomorrow, 4pm"."""
    if not old or not new:
        return new or old
    day, clock = _DAY_RE.search(new), _CLOCK_RE.search(new)
    old_day, old_clock = _DAY_RE.search(old), _CLOCK_RE.search(old)
    if day and not clock and old_clock:
        return f"{new}, {old_clock.group(0)}"
    if clock and not day and old_day:
        return f"{old_day.group(0)}, {new}"
    return new


def is_clock_time(text: str) -> bool:
    """True for a time of day alone ("4pm", "around 6 pm", "evening"): a time to call, never an installation
    timeline. In the T12a replay "4pm" (meant for the callback) was saved as the timeline and the callback was lost.
    T13e: run on the extractor's timeline value (a check on the AI's output), and on the raw message only as the backup
    when the understand step failed."""
    return bool(_CLOCK_ONLY_RE.match(text))


def ruled_out_details(message: str) -> list[str]:
    """Backup only (T13e: the understand step's ruled_out field decides; this runs when that step failed): details
    the customer rules out in a message by their words: "anything else apart from timing?" -> ["timeline"]."""
    return [_DETAIL_OF_WORD[word.lower()] for word in _EXCEPT_RE.findall(message) if word.lower() in _DETAIL_OF_WORD]


def package_case(state: dict) -> str | None:
    """What the reply does about packages on this turn (T13b). KB Sections 8 and 9 describe the packages by
    household use and give no bill ranges, so a bill alone never picks one (C7 got HomeStart 3 locally and
    HomePro 5 on Render for the same message).

    On a turn that gives the monthly bill, or the property type after it, while no system size is settled:
    "home" or "business" (name every reference package for it) or "unknown" (ask whether it is a home or a
    business first). None on any other turn, in a support case and on a contact request (name and phone first)."""
    lead = state.get("lead", {})
    if not lead.get("monthly_bill_inr") or lead.get("system_size_kw"):
        return None
    if state.get("support_case") or state.get("contact_request"):
        return None
    if not PACKAGE_DETAILS & set(state.get("new_details", [])):
        return None
    return PROPERTY_OF.get(lead.get("customer_type"), "unknown")


def next_detail(state: dict, useful: list[str]) -> str | None:
    """The one detail to ask for next (T12a): the most useful one the customer has not ruled out in this conversation,
    so the reply never asks for something already given or ruled out ("apart from timing" got a timeline question in
    the replay). T13e: a detail the customer asks about by name is the model's call (it reads the message and
    ASKED_WHAT_LINE); no word list picks it."""
    ruled_out = set(state.get("ruled_out", []))
    return next((d for d in useful if d not in ruled_out), None)


def fix_callback_claims(reply: str, state: dict) -> tuple[str, bool]:
    """Keep what a reply says about a callback true (T12a). For a sentence saying a callback was logged, updated,
    booked or confirmed:
    - schedule_callback wrote this turn: it becomes the true sentence ("I've logged a callback request for 4pm; the
      team will confirm it", or "updated ... to ...");
    - an earlier request exists: kept, but "booked", "confirmed" or "updated" becomes "is logged";
    - no callback exists: removed, and if no question is left the reply asks what time suits them.
    In the replay the reply said "I have logged a callback request for 4 PM" after schedule_callback refused.
    Returns (reply, changed)."""
    status, now, logged = state.get("callback_status"), state.get("callback_now"), state.get("callback_logged")
    lines, changed, confirmed = [], False, False
    for line in reply.split("\n"):
        kept = []
        for piece in split_sentences(line):
            text = without_tags(piece)
            if not (text and not text.endswith("?") and _CALLBACK_WORD_RE.search(text)
                    and _CALLBACK_DONE_RE.search(text)):
                if piece.strip():
                    kept.append(piece.strip())
                continue
            if status and now:  # written this turn: say exactly what was done, once
                truth = (welcome.CALLBACK_UPDATED_TEXT if now == "updated" else welcome.CALLBACK_LOGGED_TEXT)
                truth = truth.format(time=logged)
                if not confirmed:
                    kept.append(truth)
                    confirmed = True
                changed = changed or piece.strip() != truth
            elif status and _CALLBACK_FIRM_RE.search(text):
                kept.append(welcome.CALLBACK_EXISTS_TEXT.format(time=logged))
                changed = True
            elif status:
                kept.append(piece.strip())
            else:
                changed = True
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    if not changed:
        return reply, False
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if not status and "?" not in text and not is_explorer(state):  # T13e: an explorer is never asked for a time
        text = f"{text} {welcome.CALLBACK_TIME_ASK}".strip()
    return text, True


def _customer_words(item: str, state: dict) -> str:
    """A missing item in everyday words (T13e: a ruled-out phone asks for the email, and the reverse)."""
    if item == "phone or email" and "phone number" in state.get("ruled_out", []):
        return "email address"
    return welcome.CUSTOMER_WORDS[item]


def _still_needed_ask(state: dict) -> str:
    """The honest question that replaces a false "saved" claim: what the lead (or support request) still needs, one
    detail at a time (T13e; only name and phone may go together)."""
    lead = state.get("lead", {})
    if state.get("support_case"):
        missing = support_missing(lead, state.get("support", {}))
        template = welcome.SUPPORT_STILL_NEEDED_ASK
    else:
        missing = ordered_missing(lead, state.get("contact_request", False))
        template = welcome.STILL_NEEDED_ASK
    if not missing:
        return welcome.READY_TO_PASS_ASK
    return template.format(items=" and ".join(_customer_words(item, state) for item in ask_items(missing)))


def remove_false_claims(reply: str, state: dict) -> tuple[str, bool]:
    """Remove every sentence that says details were saved, recorded, logged or passed to the team when the
    turn ended with no lead and no support request (T12a), and say honestly what is still needed instead
    (only if no question is left, so the reply still asks one question at most). Returns (reply, changed).

    The prompts forbid these claims; this is the safety net for when the wording slips.
    """
    if state.get("lead_id") or state.get("support_request_id"):
        return reply, False
    lines, changed = [], False
    for line in reply.split("\n"):
        kept = []
        for piece in split_sentences(line):
            if without_tags(piece) and is_claim(piece):
                changed = True
                continue
            if piece.strip():
                kept.append(piece.strip())
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    if not changed:
        return reply, False
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if "?" not in text and not asks_parked(state):
        text = f"{text} {_still_needed_ask(state)}".strip()
    return text, True


def drop_known_city_question(reply: str, state: dict) -> tuple[str, list[str]]:
    """Never ask again for a city the bot already has (T13a): with a city in the draft, a question sentence that asks
    for the city ("Which city is your home in?") or confirms it ("Is the property in Lucknow?") is removed. If no
    question is left and the lead isn't saved yet, the honest ask for what is still needed takes its place.
    Returns (reply, the removed sentences); the graph puts them in a guard step, so a removal can be checked."""
    city = state.get("lead", {}).get("city")
    if not city or state.get("support_case"):
        return reply, []
    confirms = _confirms_city_re(str(city))
    lines, removed = [], []
    for line in reply.split("\n"):
        kept, dropped_last = [], False
        for piece in split_sentences(line):
            text = without_tags(piece)
            if not text:  # tags alone follow the sentence before them
                if piece.strip() and not dropped_last:
                    kept.append(piece.strip())
                continue
            dropped_last = text.endswith("?") and bool(_CITY_ASK_RE.search(text) or confirms.search(text))
            if dropped_last:
                removed.append(text)
            elif piece.strip():
                kept.append(piece.strip())
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    if not removed:
        return reply, []
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if "?" not in text and not state.get("lead_id") and shows_interest(state) and not asks_parked(state):
        if ordered_missing(state.get("lead", {}), state.get("contact_request", False)):
            text = f"{text} {_still_needed_ask(state)}".strip()
    logger.warning("reply asked for a city the bot already has; removed", extra={"removed": len(removed)})
    return text, removed


def drop_detail_questions(reply: str, state: dict) -> tuple[str, list[str]]:
    """While asks are parked (the customer skipped two questions about their details in a row, T13d), remove every
    question sentence that asks for a detail the assistant still needs; tags alone go with their sentence. Nothing is
    added in its place. Returns (reply, the removed sentences); the graph puts them in a guard step."""
    if not asks_parked(state):
        return reply, []
    targets = ask_targets(state)
    lines, removed = [], []
    for line in reply.split("\n"):
        kept, dropped_last = [], False
        for piece in split_sentences(line):
            text = without_tags(piece)
            if not text:  # tags alone follow the sentence before them
                if piece.strip() and not dropped_last:
                    kept.append(piece.strip())
                continue
            dropped_last = text.endswith("?") and bool(asked_items(text, targets))
            if dropped_last:
                removed.append(text)
            elif piece.strip():
                kept.append(piece.strip())
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    if not removed:
        return reply, []
    logger.warning("reply asked for details after two skipped questions; removed", extra={"removed": len(removed)})
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), removed


def save_ready_lead(state: dict, ops: LeadOps) -> dict:
    """Save the lead in code when nothing required is missing and the model didn't (T13d): no lead yet, no support case,
    and the customer did not say no in this message. Also logs a callback time they already gave. The create_lead tool
    stays the normal path; this runs in finalize, so a turn whose model failed or skipped the tool (audit S22, Render
    12:03:32 UTC) still keeps the lead, and the service's one-lead-per-session rule means it can never save twice.
    A database error is logged and nothing is saved (so no reply claims it); the next turn tries again.
    Returns the state update ({} when nothing was saved)."""
    lead = state.get("lead", {})
    if (not lead or state.get("lead_id") or state.get("support_case") or state.get("declines")
            or ordered_missing(lead, state.get("contact_request", False))):
        return {}
    try:
        areas = ops.areas_fn()
    except Exception:  # the service then marks the location "unknown"
        areas = None
    time_given = state.get("callback_time")
    try:
        result = service.create_lead(ops.engine, lead, session_id=state.get("session_id") or None, llm=ops.reason_llm,
                                     areas=areas, reason_now=ops.reason_now, callback_time=time_given,
                                     call_requested=bool(state.get("call_requested")))
    except DBAPIError as exc:
        logger.warning("lead save in code failed: %s", type(exc).__name__)
        return {}
    ops.calls.append("create")
    if result.lead is None:
        return {}
    update = {"lead_id": result.lead.id, "lead_saved": dict(lead), "lead_created_now": True}
    if result.lead.reason_source == "template" and not ops.reason_now:
        update["reason_pending"] = result.lead.id
    if time_given and time_given != state.get("callback_logged"):
        try:
            updated = service.schedule_callback(ops.engine, result.lead.id, time_given).updated
        except DBAPIError as exc:  # the lead is saved; the tool can log the callback on a later turn
            logger.warning("callback log in code failed: %s", type(exc).__name__)
        else:
            ops.calls.append("callback")
            update |= {"callback_status": "requested", "callback_logged": time_given,
                       "callback_now": "updated" if updated else "requested"}
    logger.info("lead saved in code (create_lead was not called)", extra={"lead_id": result.lead.id})
    return update


def add_saved_line(reply: str, outcome: str) -> str:
    """The reply of a turn whose lead was saved in code after the model answered (T13d): the model's questions go (they
    asked for details the lead already has, e.g. S22's "could you share your phone number so we can save your
    details?") and `outcome` (what was saved, with its own question) closes the reply."""
    lines = []
    for line in reply.split("\n"):
        kept, dropped_last = [], False
        for piece in split_sentences(line):
            text = without_tags(piece)
            if not text:
                if piece.strip() and not dropped_last:
                    kept.append(piece.strip())
                continue
            dropped_last = text.endswith("?")
            if not dropped_last and piece.strip():
                kept.append(piece.strip())
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return f"{text}\n\n{outcome}" if text else outcome


def _wanted_item(state: dict) -> str | None:
    """The detail the reply should ask for next, if any: whether the property is a home or a business on a bill turn
    that doesn't know it yet (T13b), the support request's next need, the lead's next missing field once the
    customer showed interest, or after the save the next useful detail."""
    lead = state.get("lead", {})
    if package_case(state) == "unknown":
        return "property type"
    if asks_parked(state):  # T13d: they skipped two questions in a row
        return None
    if region_turn(state) and not state.get("support_case"):  # T13e: a region named; which city is needed to answer
        return "city"
    if state.get("support_case") and not state.get("support_request_id"):
        missing = support_missing(lead, state.get("support", {}))
        return missing[0] if missing else None
    if not state.get("lead_id"):
        missing = ordered_missing(lead, state.get("contact_request", False))
        return missing[0] if missing and shows_interest(state) else None
    ruled_out = set(state.get("ruled_out", []))
    useful = [d for d in missing_details(LeadData.model_validate(lead), _breakdown_stub(lead)) if d not in ruled_out]
    return next_detail(state, useful) if useful else None


def keep_one_question(reply: str, state: dict) -> tuple[str, bool]:
    """At most one question per reply (T12b; the system prompt's rule, which Groq broke in the T12a drill). With
    several, keep the one that asks for the next missing detail, else the last one (usually the closing offer:
    "Is that helpful? Would you like a callback?" keeps the callback question). Returns (reply, changed)."""
    pieces = [(i, j, piece) for i, line in enumerate(reply.split("\n"))
              for j, piece in enumerate(split_sentences(line))]
    questions = [(i, j) for i, j, piece in pieces if without_tags(piece).endswith("?")]
    if len(questions) < 2:
        return reply, False
    item = _wanted_item(state)
    words = _ITEM_WORDS.get(item, ()) if item else ()
    asking = [(i, j) for i, j, piece in pieces if (i, j) in questions
              and any(re.search(rf"\b{w}\b", without_tags(piece), re.I) for w in words)]
    keep = asking[0] if asking else questions[-1]
    lines: dict[int, list[str]] = {}
    dropped_last = False
    for i, j, piece in pieces:
        lines.setdefault(i, [])
        bare = without_tags(piece)
        if (i, j) in questions and (i, j) != keep:
            dropped_last = True
            continue
        if not bare and dropped_last:  # tags of a dropped question go with it
            continue
        dropped_last = False
        if piece.strip():
            lines[i].append(piece.strip())
    original = reply.split("\n")
    text = "\n".join(re.match(r"[ \t]*", original[i]).group(0) + " ".join(kept) if kept else ""
                     for i, kept in sorted(lines.items()))
    logger.warning("reply asked more than one question; kept one", extra={"questions": len(questions)})
    return re.sub(r"\n{3,}", "\n\n", text).strip(), True


def _saved_lead_next(state: dict, useful: list[str]) -> str:
    """What the agent should do next once the lead is saved (T12a)."""
    intents = set(state.get("intents", []))
    if state.get("callback_now"):
        return CALLBACK_NOW_LINE
    if state.get("lead_created_now"):
        if state.get("callback_time"):
            return SAVED_WITH_TIME_LINE
        return SAVED_EXPLORING_LINE if is_explorer(state) else SAVED_NOW_LINE
    if not useful:
        return NOTHING_MORE_LINE
    if state.get("declines"):
        return SAVED_INFO_LINE
    detail = next_detail(state, useful)
    if "asks_what_is_needed" in intents:
        return ASKED_WHAT_LINE.format(detail=detail) if detail else NOTHING_LEFT_LINE
    if (intents & DETAIL_INTENTS or state.get("new_details")) and detail:
        return INTEREST_LINE.format(detail=detail)
    return SAVED_INFO_LINE


def status_block(state: dict, offers_made: list[str] = ()) -> str:
    """The lead status the agent reads each turn: field names and plain messages, never values,
    never a score or a status. T13e: `offers_made` are the kinds of next step the assistant already offered in this
    chat (app/agent/questions.py), never to be offered again."""
    lead = state.get("lead", {})
    lines = []
    if state.get("support_case"):
        missing = support_missing(lead, state.get("support", {}))
        if state.get("support_request_id"):
            lines.append(f"Support request saved (id {state['support_request_id']}). Do not create a sales lead.")
        elif missing and asks_parked(state):  # T13d
            lines.append("This is an existing customer's issue (no sales lead). " + SKIPPED_LINE)
        elif missing:
            lines.append("This is an existing customer's issue (no sales lead). Still needed for the support "
                         "request: " + ", ".join(missing) + ".")
        else:
            lines.append("Ready to save: call create_support_request now.")
    else:
        known = sorted(lead)
        lines.append("Details known: " + (", ".join(known) if known else "none yet") + ".")
        if state.get("lead_id"):
            lines.append("Lead saved: yes.")
            ruled_out = set(state.get("ruled_out", []))  # "apart from timing": not asked in this conversation
            useful = [d for d in missing_details(LeadData.model_validate(lead), _breakdown_stub(lead))
                      if d not in ruled_out]
            if useful and not asks_parked(state):
                lines.append("Would still add points, most useful first: " + ", ".join(useful)
                             + ". Never ask again for anything under Details known.")
            lines.append(SKIPPED_LINE if asks_parked(state) and not state.get("callback_now")
                         and not state.get("lead_created_now") else _saved_lead_next(state, useful))
        else:
            missing = ordered_missing(lead, state.get("contact_request", False))
            if missing and not shows_interest(state):  # a pure information question: no details asked (T10b)
                lines.append(INFO_ONLY_LINE)
            elif missing and asks_parked(state):  # T13d
                lines.append("Lead saved: no, so never say the details are saved, recorded or passed to the team. "
                             + SKIPPED_LINE)
            elif missing:
                lines.append("Lead saved: no, so never say the details are saved, recorded or passed to the team. "
                             "Still needed before it can be saved: " + ", ".join(missing) + ".")
            else:
                lines.append("Lead saved: no. Ready to save: call create_lead now (sharing a phone number "
                             "or email counts as wanting follow-up).")
    if state.get("contact_request") and not state.get("lead_id"):
        lines.append("The customer asked to be contacted: ask for what is still needed (name and phone first, in one "
                     "question); do not answer earlier questions again.")
    if offers_made:
        lines.append("Offers already made in this chat: " + ", ".join(offers_made) + ". Do not make them again.")
    if state.get("problem_unclear") and not state.get("support_case"):
        lines.append(PROBLEM_UNCLEAR_LINE)
    for name, message in sorted(state.get("validation_errors", {}).items()):
        lines.append(f"Looked wrong, ask again: {name.replace('_', ' ')}: {message}")
    if not state.get("support_case"):
        if lead.get("city") and state.get("city_source") == "asked":  # T13a
            lines.append(ASKED_CITY_LINE)
        cities = state.get("asked_cities", [])
        missing = ordered_missing(lead, state.get("contact_request", False)) if not state.get("lead_id") else []
        if len(cities) > 1 and not lead.get("city") and missing[:1] == ["city"]:  # only when the city is next
            lines.append("Cities asked about: " + ", ".join(cities) + ". Ask which one the property is in.")
    lines.extend(_callback_lines(state))
    return "\n".join(lines)


def _callback_lines(state: dict) -> list[str]:
    """The callback part of the status block. A lead keeps one requested callback, so a new time updates it."""
    time_given, logged = state.get("callback_time"), state.get("callback_logged")
    if state.get("lead_id") and time_given and time_given != logged:
        if state.get("callback_status"):
            return ["The customer gave a new callback time: call schedule_callback to update the request."]
        return ["The customer gave a preferred callback time: call schedule_callback."]
    if state.get("callback_status"):
        return [f"Callback request logged ({state['callback_status']}); the team will confirm it. If the "
                "customer gives a new time, call schedule_callback again: it updates the request."]
    if time_given:
        return ["The customer gave a preferred callback time."]
    if state.get("lead_id") and state.get("contact_request"):
        return ["The customer asked to be contacted: ask what time suits them for a call."]
    return []


def _breakdown_stub(lead: dict) -> dict:
    """missing_details only reads the requirement and property rule values; build them from the draft."""
    return score_lead(LeadData.model_validate(lead), "unknown")


def card_fields(values: dict) -> dict:
    """What the chat page's sales card shows: the name, city and type, plus the names of the fields that
    are known (only names, so no phone number, email or money amount is ever sent)."""
    return {
        "name": values.get("name"),
        "city": values.get("city"),
        "property_type": values.get("property_type"),
        "customer_type": values.get("customer_type"),
        "known_fields": sorted(name for name, value in values.items() if value is not None),
    }


def lead_summary(state: dict, engine: Engine | None) -> dict | None:
    """{id, status, score, missing_fields, callback_status, name, city, property_type, customer_type,
    known_fields} for the API's demo panel, or None when nothing about a lead is known. The API hides
    status and score unless CHAT_SHOW_LEAD_SCORE is on."""
    lead = state.get("lead", {})
    if state.get("support_case") and not state.get("lead_id"):
        return None
    if state.get("lead_id") and engine is not None:
        view = service.get_lead(engine, state["lead_id"])
        if view is not None:
            callbacks = view.callbacks
            return {
                "id": view.id,
                "status": view.lead_status,
                "score": view.lead_score,
                "missing_fields": missing_details(view, view.score_breakdown),
                "callback_status": callbacks[-1].status if callbacks else None,
                **card_fields(view.model_dump(include=set(LeadData.model_fields))),
            }
    if not lead:
        return None
    return {
        "id": None,
        "status": None,
        "score": None,
        "missing_fields": ordered_missing(lead, state.get("contact_request", False)),
        "callback_status": None,
        **card_fields(LeadData.model_validate(lead).model_dump()),
    }

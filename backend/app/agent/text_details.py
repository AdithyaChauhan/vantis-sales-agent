"""What code reads from the customer's own words (T13d), without a model.

- details_from_text: when the understand step fails (every model slow or down), the phone, email, city, monthly bill,
  system size and, only from clear patterns, the name. In the Render logs of 13 Oct the turn "main Rakesh Sharma hu,
  mera number 9876543210 hai, email …" failed on every model, so nothing was stored and no lead was saved.
- is_what_if / reports_problem_now: "agar inverter kharab ho jaye to kya karna padega" asks what to do if it breaks; it
  is a question, not a fault to raise a support request for.
- is_go_ahead: a clear new buying signal ("I want to go ahead", "how do I book", "lagwana hai"), after which the
  assistant may ask for details again (leads.py, skipped asks).
- asks_to_be_called: "can someone call me?", "mujhe call karo": a contact request, read in code when the understand
  step fails, so it still counts as ready intent (live check of 13 Oct: Google timed out on every understand call).
- asks_what_is_offered: "What solar systems do you provide?", "kya kya milta hai": the KB Section 8 list joins the
  context with or without the understand step (on 13 Oct the raw search for the assignment's Q1 found Section 1).
- details_from_text also reads clear intent and timeline phrases ("just exploring", "next year", "agle mahine"), so
  "just exploring, maybe next year" ends COLD on a day the understand step fails.

Every value still goes through LeadData's validators, so a bad phone or email becomes a question, never a stored value.
"""
import re

from app.agent.packages import SIZE_RE
from app.services.service_areas import ServiceAreas

# A phone-like run of 8-13 digits (spaces, dashes, a +91 allowed); clean_phone decides whether it is a mobile number
_PHONE_RE = re.compile(r"(?<![\w+])\+?\d(?:[\s-]?\d){7,12}(?!\d)")
_EMAIL_RE = re.compile(r"[^\s@,;<>()\[\]]+@[^\s@,;<>()\[\]]+")
# A number next to "bill": "bill is about ₹4,000", "bijli bill 3500 aata hai", "4000 ka bill", "3500 monthly bill"
_AMOUNT = r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?\s*(?:k|thousand|lakh|lac)?)\b"
_BILL_AFTER_RE = re.compile(rf"\bbill\b\D{{0,25}}?{_AMOUNT}", re.I)
_BILL_BEFORE_RE = re.compile(rf"{_AMOUNT}\s*(?:rs\.?|rupees?|rupaye|/-)?\s*(?:ka|ki|monthly|per\s+month|a\s+month)?\s*"
                             r"(?:bijli\s+|electricity\s+)?bill\b", re.I)
# An unlisted city only from a clear phrase; a listed one is found by ServiceAreas.find_in_text
_CITY_PHRASE_RE = re.compile(
    r"\b(?:i\s+(?:live|stay)\s+in|(?:i|we)(?:['’]m|['’]re|\s+am|\s+are)\s+(?:from|based\s+in)|my\s+city\s+is|"
    r"(?:the\s+|my\s+|our\s+)?(?:house|home|property|shop|office|flat)\s+is\s+in|based\s+in|located\s+in)\s+"
    r"([a-z][a-z.'-]*(?:\s+[a-z][a-z.'-]*){0,2}?)(?=\s*(?:[,.!?;]|$)|\s+(?:and|but|so|with|for|my|i|we|area|city)\b)",
    re.I)
_CITY_HINDI_RE = re.compile(r"\b([a-z]+)\s+(?:mein|me|mai)\s+(?:rehta|rehti|rahta|rahti|rehte|rahte)\b|"
                            r"\bmain\s+([a-z]+)\s+se\s+(?:hu|hoon|hun|hoo)\b", re.I)
_NOT_A_PLACE = {"a", "an", "the", "my", "our", "this", "that", "here", "there", "india", "hurry", "town", "city"}
# The name, only from clear patterns (Adithya's list): "my name is X", "mera naam X hai", "main X hu", or 1-3 name-like
# words right before the phone number or email
_NAME_WORD = r"[a-z][a-z.'-]*"
_NAME_IS_RE = re.compile(rf"\bmy\s+name\s+is\s+({_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}}?)"
                         r"(?=\s*(?:[,.!?;]|$)|\s+(?:and|from|i|my|here|aur|hai)\b)", re.I)
_MERA_NAAM_RE = re.compile(rf"\bmera\s+naam\s+({_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}}?)\s+(?:hai|h)\b", re.I)
_MAIN_HU_RE = re.compile(rf"\bmain\s+({_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}}?)\s+(?:hu|hoon|hun|hoo|hu+n)\b", re.I)
_BEFORE_CONTACT_RE = re.compile(
    rf"(?:^|[,.;:!?]\s*|\b(?:hi|hello|hey|namaste)[,!]?\s+)({_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}})\s*[,:-]?\s*"
    r"(?=\+?\d[\d\s-]{7,}|[^\s@,]+@)", re.I)
_INTRO_RE = re.compile(r"^(?:i['’]?m|i\s+am|this\s+is|it['’]?s)\s+", re.I)
# Words that are never part of a name (Hindi postpositions and fillers, contact words, everyday words)
_NOT_NAME = {
    "my", "number", "phone", "mobile", "mob", "ph", "no", "num", "email", "mail", "id", "is", "call", "me", "on", "at",
    "contact", "and", "the", "here", "its", "it's", "mera", "meri", "mere", "naam", "hai", "ka", "ki", "ke", "ko", "se",
    "mein", "me", "mai", "par", "pe", "whatsapp", "or", "to", "us", "please", "pls", "reach", "ghar", "yahan", "wahan",
    "abhi", "bhi", "toh", "ready", "interested", "bahut", "ek", "you", "your", "a", "an", "i", "am", "we", "our", "sir",
    "madam", "ji", "thanks", "thank", "okay", "ok", "yes", "haan", "solar", "bill", "kya", "kaise", "aap", "log",
}
# A what-if: "what if", "in case", "suppose", "agar", "yadi", a clause that starts with "if", "ho jaye" (would become)
_WHAT_IF_RE = re.compile(r"\b(?:what\s+if|in\s+case|suppose|supposing|what\s+happens\s+(?:if|when)|"
                         r"what\s+would\s+happen|agar|yadi)\b|(?:^|[.,!?;]\s*)if\b|\bho\s+ja(?:a)?(?:ye|e)\b", re.I)
# A problem the customer has now
_PROBLEM_NOW_RE = re.compile(
    r"\b(?:(?:is|isn['’]?t|not|stopped|has\s+stopped)\s+working|stopped|is\s+(?:broken|down|faulty|dead)|"
    r"not\s+(?:generating|charging|turning\s+on|starting)|showing\s+(?:an?\s+)?error|"
    r"(?:kharab|band|kharaab)\s+ho\s+(?:gaya|gayi|gya|gyi|chuka|chuki)|kaam\s+nahi\s+kar\s+(?:raha|rahi|rhe|rahe)|"
    r"chal\s+nahi\s+(?:raha|rahi))\b", re.I)
# A clear new buying signal (Adithya: "I want to go ahead", "how do I book", "lagwana hai, aage kya karna hai")
_GO_AHEAD_RE = re.compile(
    r"\b(?:go\s+ahead|proceed|book(?:ing)?\b|sign\s+up|get\s+started|start\s+the\s+process|next\s+steps?|"
    r"how\s+(?:do|can)\s+i\s+(?:buy|order|start|get\s+(?:it|one|solar))|"
    r"(?:want|would\s+like|ready|planning)\s+to\s+(?:buy|install|order|go\s+ahead|get\s+(?:it|solar|one))|"
    r"lagwana\s+(?:hai|chahta|chahti|chahte)|lagwani\s+hai|aage\s+kya)", re.I)


# Only exploring, and when to install, from clear phrases (Adithya's lists); "jaldi" only when it isn't about a reply
_EXPLORING_RE = re.compile(r"\bjust\s+(?:exploring|looking|browsing|checking)\b|\bonly\s+(?:exploring|looking)\b|"
                           r"\bnot\s+now\b|\b(?:abhi\s+)?bas\s+dekh\s+(?:raha|rahi|rahe)\b", re.I)
_TIMELINES = (  # (bucket, phrases), checked in this order; "next month" is within about a month, as in SCORING.md
    ("gt_6m", re.compile(r"\bnext\s+year\b|\bagle\s+saal\b", re.I)),
    ("lt_1m", re.compile(r"\bthis\s+month\b|\bnext\s+month\b|\bagle\s+mahine\b|\basap\b|\bturant\b|"
                         r"\bjaldi\b(?!\s+(?:batao|bataiye|bataye|bolo|reply|jawab|answer))", re.I)),
)
# The message asks what Vantis offers (Adithya's words): its systems, packages, products, options or services
_OFFER_NOUNS = r"(?:systems?|packages?|products?|offerings?|options|services|solutions|plans)"
_OFFER_RE = re.compile(
    rf"\b(?:what|which|kaun\s?se|kaunsa|kaunse|kya\s+kya)\b[^?.!]{{0,40}}\b{_OFFER_NOUNS}\b(?!\s+areas?)|"
    rf"\b{_OFFER_NOUNS}\b[^?.!]{{0,30}}\b(?:do\s+)?you\s+(?:offer|provide|have|sell)\b|"
    r"\bkya\s+kya\s+(?:milta|milte|hai|hain|dete|offer)\b|"
    r"\bwhat\s+(?:all\s+|else\s+)?(?:do|does|can)\s+(?:you|vantis)\s+(?:offer|provide|sell)\b(?!\s+(?:in|at)\b)", re.I)
_NOT_AN_OFFER_RE = re.compile(r"\b(?:financ\w*|emi|loan|subsid\w*|warrant\w*|pay\w*|instal+ments?)\b", re.I)
# A request to be called or contacted, in English or Hinglish
_CALL_ME_RE = re.compile(
    r"\b(?:call|ring|phone|contact|reach)\s+(?:me|us)\b|\bcall\s*back\b|\bcallback\b|\bget\s+in\s+touch\b|"
    r"\b(?:someone|somebody|anyone|your\s+team)\s+(?:to\s+|can\s+|could\s+|please\s+)?(?:call|contact|reach)\b|"
    r"\b(?:mujhe|hume|humein|hamein)\s+(?:call|phone|contact)\b|\b(?:call|phone)\s+(?:karo|kariye|kijiye|karna|kar\s+do|"
    r"kar\s+dena)\b", re.I)


def _phone(message: str) -> str | None:
    match = _PHONE_RE.search(message)
    return match.group(0).strip() if match else None


def _email(message: str) -> str | None:
    match = _EMAIL_RE.search(message)
    return match.group(0).rstrip(".,!?:;") if match else None


def _bill(message: str) -> str | None:
    for pattern in (_BILL_AFTER_RE, _BILL_BEFORE_RE):
        match = pattern.search(message)
        if match:
            return match.group(1).strip()
    return None


def _size(message: str) -> str | None:
    match = SIZE_RE.search(message)
    return match.group(0) if match else None


def _city(message: str, areas: ServiceAreas | None) -> str | None:
    """A listed city (or a look-alike phrase) found in the words, else an unlisted one from a clear phrase."""
    if areas is not None:
        found = areas.find_in_text(message)
        if found:
            return found[0]
    for pattern in (_CITY_PHRASE_RE, _CITY_HINDI_RE):
        for match in pattern.finditer(message):
            place = next(g for g in match.groups() if g)
            words = place.split()
            if words and words[0].lower() not in _NOT_A_PLACE and not set(w.lower() for w in words) & _NOT_NAME:
                return " ".join(w[:1].upper() + w[1:] for w in words)
    return None


def _intent_and_timeline(message: str) -> dict:
    """intent_level "exploring" and the timeline (the words and their bucket) from clear phrases only."""
    found = {}
    if _EXPLORING_RE.search(message):
        found["intent_level"] = "exploring"
    for bucket, pattern in _TIMELINES:
        match = pattern.search(message)
        if match:
            found |= {"timeline": match.group(0), "timeline_bucket": bucket}
            break
    return found


def _looks_like_name(text: str, areas: ServiceAreas | None) -> bool:
    words = [w.lower() for w in text.split()]
    if not words or set(words) & _NOT_NAME or any(w[0].isdigit() for w in words):
        return False
    return not (areas is not None and areas.find_in_text(text))  # "main Lucknow se hu" is not a name


def _name(message: str, areas: ServiceAreas | None) -> str | None:
    for pattern in (_NAME_IS_RE, _MERA_NAAM_RE, _MAIN_HU_RE, _BEFORE_CONTACT_RE):
        for match in pattern.finditer(message):
            name = " ".join(match.group(1).split()).strip(".'-")
            if pattern is _BEFORE_CONTACT_RE:  # "I'm Vikram Rao, 98765…": the name is what follows "I'm"
                name = _INTRO_RE.sub("", name)
            if _looks_like_name(name, areas):
                return name.title() if name.islower() else name
    return None


def details_from_text(message: str, areas: ServiceAreas | None = None) -> dict:
    """The lead details code can read from a message, as raw values for LeadData.from_raw (which validates them):
    phone, email, city, monthly_bill_inr, system_size_kw, name, and from clear phrases intent_level "exploring" and the
    timeline with its bucket. Only what is found is returned."""
    found = {
        "phone": _phone(message),
        "email": _email(message),
        "city": _city(message, areas),
        "monthly_bill_inr": _bill(message),
        "system_size_kw": _size(message),
        "name": _name(message, areas),
        **_intent_and_timeline(message),
    }
    return {key: value for key, value in found.items() if value}


def is_what_if(message: str) -> bool:
    """True for a question about something that might happen ("agar inverter kharab ho jaye to kya karna padega",
    "what if the panels get damaged?")."""
    return bool(_WHAT_IF_RE.search(message))


def reports_problem_now(message: str) -> bool:
    """True when the message says something is wrong now ("my inverter stopped working", "kharab ho gaya hai")."""
    return bool(_PROBLEM_NOW_RE.search(message))


def asks_what_is_offered(message: str) -> bool:
    """True when the message asks what Vantis offers ("What solar systems do you provide?", "kaunse system hain",
    "kya kya milta hai"), but not a question about financing, subsidies, warranty or payment options."""
    return bool(_OFFER_RE.search(message)) and not _NOT_AN_OFFER_RE.search(message)


def asks_to_be_called(message: str) -> bool:
    """True for a request to be called or contacted ("can someone call me?", "please call back", "mujhe call karo")."""
    return bool(_CALL_ME_RE.search(message))


def is_go_ahead(message: str) -> bool:
    """True for a clear new buying signal ("I want to go ahead", "how do I book a site visit", "lagwana hai")."""
    return bool(_GO_AHEAD_RE.search(message))

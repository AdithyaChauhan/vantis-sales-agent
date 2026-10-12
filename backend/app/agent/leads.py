"""Lead state for the agent: the status block the model reads, city confirmation, the write-once sync rule
and the lead summary for the API. Calls the T7 leads service; never decides scores itself.

The model never sees a score, a status, a reason or any personal value: the status block lists field
names and plain messages only.
"""
import re
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from sqlalchemy import Engine

from app.leads import service
from app.leads.schemas import LeadData
from app.leads.scoring import missing_details, missing_for_creation, score_lead
from app.services.service_areas import ServiceAreas

# The order F2 asks for what is missing; a contact request puts name and phone first
NEED_ORDER = ["city", "requirement, monthly bill, usage, system size or property type", "name", "phone or email"]
CONTACT_ORDER = ["name", "phone or email", "city", "requirement, monthly bill, usage, system size or property type"]
SUPPORT_NEEDS = ("name", "phone or email", "issue", "project details")
BUYING_INTENTS = {"buying_interest", "contact_request"}
INFO_ONLY_LINE = ("Lead saved: no. The customer has only asked for information so far: do not ask for their "
                  "details; after answering, offer one next step.")
_QUESTION_RE = re.compile(r"[^.!?]*\?")


@dataclass
class LeadOps:
    """What the lead tools and the finalize sync use: the leads database and the reason model.

    Live tests pass the Neon test branch here, so they never write to the main database.
    """

    engine: Engine
    areas_fn: Callable[[], ServiceAreas]
    reason_llm: Runnable | None = None  # None: the service uses fast_llm()
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


def shows_interest(state: dict) -> bool:
    """The customer wants solar for their own place: a buying or contact intent this turn, a city they asked
    about, or any detail about themselves in the draft (intent_level alone is the extractor's guess)."""
    if set(state.get("intents", [])) & BUYING_INTENTS or state.get("asked_cities"):
        return True
    return any(name != "intent_level" for name in state.get("lead", {}))


def pending_changes(lead: dict, saved: dict) -> dict:
    """Draft fields that differ from what was last written to the database."""
    return {k: v for k, v in lead.items() if saved.get(k) != v}


def final_question(message) -> str:
    """The last question sentence of an assistant message ('' if it asked none)."""
    if not isinstance(message, AIMessage):
        return ""
    questions = _QUESTION_RE.findall(str(message.content))
    return questions[-1] if questions else ""


def confirmed_city(asked_cities: list[str], previous: object, out: dict) -> str | None:
    """A likely city the customer confirmed: the assistant's last question named it, and the reply
    gives no other city, isn't off-topic and doesn't say no. None otherwise."""
    if not asked_cities or out["details"].get("city") or out.get("declines"):
        return None
    if "off_topic" in out["intents"]:
        return None
    question = final_question(previous).lower()
    for city in reversed(asked_cities):
        if city.lower() in question:
            return city
    return None


def status_block(state: dict) -> str:
    """The lead status the agent reads each turn: field names and plain messages, never values,
    never a score or a status."""
    lead = state.get("lead", {})
    lines = []
    if state.get("support_case"):
        missing = support_missing(lead, state.get("support", {}))
        if state.get("support_request_id"):
            lines.append(f"Support request saved (id {state['support_request_id']}). Do not create a sales lead.")
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
            if not state.get("callback_status"):
                lines.append("Next: say the sales team can follow up and offer a callback (ask what time suits "
                             "them). Do not ask for more details in the same turn.")
            else:
                useful = missing_details(LeadData.model_validate(lead), _breakdown_stub(lead))
                if useful:
                    lines.append("Still useful, ask gently only if it fits: " + ", ".join(useful) + ".")
        else:
            missing = ordered_missing(lead, state.get("contact_request", False))
            if missing and not shows_interest(state):  # a pure information question: no details asked (T10b)
                lines.append(INFO_ONLY_LINE)
            elif missing:
                lines.append("Lead saved: no. Still needed before it can be saved: " + ", ".join(missing) + ".")
            else:
                lines.append("Lead saved: no. Ready to save: call create_lead now (sharing a phone number "
                             "or email counts as wanting follow-up).")
    if state.get("contact_request"):
        lines.append("The customer asked to be contacted: ask for what is still needed (name and phone first); "
                     "do not answer earlier questions again.")
    for name, message in sorted(state.get("validation_errors", {}).items()):
        lines.append(f"Looked wrong, ask again: {name.replace('_', ' ')}: {message}")
    cities = state.get("asked_cities", [])
    if cities and not lead.get("city") and not state.get("support_case"):
        if len(cities) == 1:
            lines.append(f"City asked about, not confirmed as theirs: {cities[0]}. Confirm it inside your next "
                         "question.")
        else:
            lines.append("Cities asked about: " + ", ".join(cities) + ". Ask which one the property is in.")
    if state.get("callback_status"):
        lines.append(f"Callback request logged ({state['callback_status']}); the team will confirm it.")
    elif state.get("callback_time"):
        lines.append("The customer gave a preferred callback time.")
    return "\n".join(lines)


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

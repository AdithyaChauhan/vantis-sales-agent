"""Lead service: the functions T8's agent tools and T9's API call.

Every function takes the engine as an argument; the LLM and the service-area list can be passed in
(tests use fakes). Logs carry ids, status and score only: never names, phone numbers or emails.
"""
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable
from sqlalchemy import Engine, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.llm import fast_llm
from app.leads import validators as v
from app.leads.models import Callback, Lead, SupportRequest
from app.leads.reason import template_reason, write_reason
from app.leads.schemas import (
    CallbackResult,
    CallbackView,
    CreateLeadResult,
    LeadData,
    LeadView,
    SupportRequestResult,
    SupportRequestView,
    UpdateLeadResult,
)
from app.leads.scoring import missing_for_creation, recommended_action, score_lead
from app.services.service_areas import ServiceAreas, get_service_areas

logger = logging.getLogger(__name__)

CALLBACK_MESSAGE = "Callback request logged; the team will confirm."
STATUSES = ("HOT", "WARM", "COLD")
CHANNELS = ("web", "whatsapp", "demo")  # demo: scripts/seed_demo_leads.py
SORTS = ("newest", "score")
MAX_LIMIT = 200
PREFERRED_TIME_CHARS = 200


class LeadNotFound(LookupError):
    """No lead with that id."""


@dataclass
class Location:
    """Where the customer is: city (matched service city when listed), their own words, and the status."""

    city: str | None
    city_input: str | None
    status: str  # listed | unverified | unknown


def locate(city: str | None, state: str | None, areas: ServiceAreas | None) -> Location:
    """Check the city with the existing service-area lookup (KB Section 10). No list: 'unknown'."""
    if not city:
        return Location(None, None, "unknown")
    if areas is None:
        return Location(city, city, "unknown")
    result = areas.lookup(city, state)
    return Location(result.matched_city or city, city, result.status)


def _areas(areas: ServiceAreas | None) -> ServiceAreas | None:
    if areas is not None:
        return areas
    try:
        return get_service_areas()
    except Exception as exc:  # DB down or not ingested: the lead is still saved, location 'unknown'
        logger.warning("service areas unavailable: %s", type(exc).__name__)
        return None


def _lead_data(row: Lead) -> LeadData:
    return LeadData.model_validate({name: getattr(row, name) for name in LeadData.model_fields})


def _requested_callback_time(row: Lead) -> str | None:
    requested = [c for c in row.callbacks if c.status == "requested"]
    return requested[-1].preferred_time if requested else None


def _log(message: str, row: Lead) -> None:
    logger.info(
        message, extra={"lead_id": row.id, "lead_status": row.lead_status, "lead_score": row.lead_score}
    )


def create_lead(
    engine: Engine,
    raw: dict[str, Any],
    *,
    session_id: str | None = None,
    channel: str = "web",
    llm: Runnable | None = None,
    areas: ServiceAreas | None = None,
    reason_now: bool = True,
) -> CreateLeadResult:
    """Validate, score and save a lead. Saves nothing (and lists what is missing) until the minimum is there.

    A second call with the same session_id updates that session's lead instead of creating another, even
    when two calls race (the UNIQUE session_id constraint decides). reason_now=False saves the template
    reason, so a caller can have the LLM write it after replying (refresh_reason).
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}")
    if session_id:
        with Session(engine) as session:
            existing_id = session.scalar(select(Lead.id).where(Lead.session_id == session_id))
        if existing_id is not None:
            updated = update_lead(engine, existing_id, raw, llm=llm, areas=areas)
            return CreateLeadResult(lead=updated.lead, errors=updated.errors)

    data, errors = LeadData.from_raw(raw)
    missing = missing_for_creation(data)
    if missing:
        logger.info("lead not created: %d required details missing", len(missing))
        return CreateLeadResult(missing_fields=missing, errors=errors)

    location = locate(data.city, data.state, _areas(areas))
    data = data.model_copy(update={"city": location.city})
    breakdown = score_lead(data, location.status)
    if reason_now:
        reason, reason_source = write_reason(breakdown, data.city, llm or fast_llm())  # before the DB write
    else:
        reason, reason_source = template_reason(breakdown), "template"

    row = Lead(
        **data.model_dump(),
        city_input=location.city_input,
        session_id=session_id,
        source_channel=channel,
        location_status=location.status,
        lead_score=breakdown["lead_score"],
        lead_status=breakdown["lead_status"],
        score_breakdown=breakdown,
        reason=reason,
        reason_source=reason_source,
        recommended_action=recommended_action(breakdown, data),
    )
    try:
        with Session(engine, expire_on_commit=False) as session, session.begin():
            session.add(row)
            session.flush()
            session.refresh(row)
            view = LeadView.model_validate(row)
    except IntegrityError:
        if not session_id:
            raise
        # Another request for this session saved its lead first: update that one instead
        logger.info("lead create raced for one session; updating the existing lead")
        with Session(engine) as session:
            existing_id = session.scalar(select(Lead.id).where(Lead.session_id == session_id))
        updated = update_lead(engine, existing_id, raw, llm=llm, areas=areas)
        return CreateLeadResult(lead=updated.lead, errors=updated.errors)
    _log("lead created", row)
    return CreateLeadResult(lead=view, errors=errors)


def update_lead(
    engine: Engine,
    lead_id: int,
    raw: dict[str, Any],
    *,
    llm: Runnable | None = None,
    areas: ServiceAreas | None = None,
    reason_now: bool = True,
) -> UpdateLeadResult:
    """Merge new valid details into a lead, re-check the city if it changed, re-score and save.

    The reason is rewritten only when the score or the status changed (saves LLM quota); with
    reason_now=False the template is saved and refresh_reason writes the LLM reason later.
    Raises LeadNotFound.
    """
    new, errors = LeadData.from_raw(raw)
    changes = new.model_dump(exclude_none=True)
    with Session(engine) as session:
        row = session.get(Lead, lead_id)
        if row is None:
            raise LeadNotFound(lead_id)
        data = _lead_data(row).model_copy(update=changes)
        city_input, location_status = row.city_input, row.location_status
        old_score, old_status = row.lead_score, row.lead_status
        reason, reason_source = row.reason, row.reason_source
        callback_time = _requested_callback_time(row)

    if "city" in changes or "state" in changes:
        location = locate(changes.get("city", city_input), data.state, _areas(areas))
        data = data.model_copy(update={"city": location.city})
        city_input, location_status = location.city_input, location.status
    breakdown = score_lead(data, location_status)
    if (breakdown["lead_score"], breakdown["lead_status"]) != (old_score, old_status):
        if reason_now:
            reason, reason_source = write_reason(breakdown, data.city, llm or fast_llm())
        else:
            reason, reason_source = template_reason(breakdown), "template"

    with Session(engine, expire_on_commit=False) as session, session.begin():
        row = session.get(Lead, lead_id)
        if row is None:
            raise LeadNotFound(lead_id)
        for name, value in data.model_dump().items():
            setattr(row, name, value)
        row.city_input = city_input
        row.location_status = location_status
        row.lead_score = breakdown["lead_score"]
        row.lead_status = breakdown["lead_status"]
        row.score_breakdown = breakdown
        row.reason, row.reason_source = reason, reason_source
        row.recommended_action = recommended_action(breakdown, data, callback_time)
        session.flush()
        session.refresh(row)
        view = LeadView.model_validate(row)
    _log("lead updated", row)
    return UpdateLeadResult(lead=view, errors=errors)


def refresh_reason(engine: Engine, lead_id: int, llm: Runnable | None = None) -> bool:
    """Replace a template reason with the LLM's sentence (run after the chat reply, T8b).

    Writes only if the lead still has the score and status the sentence was written for. Never raises
    (it runs in the background): returns True if the reason was replaced.
    """
    try:
        with Session(engine) as session:
            row = session.get(Lead, lead_id)
            if row is None or row.reason_source != "template":
                return False
            breakdown, city = row.score_breakdown, row.city
        reason, source = write_reason(breakdown, city, llm or fast_llm())
        if source != "llm":
            return False
        with Session(engine) as session, session.begin():
            row = session.get(Lead, lead_id)
            same = row is not None and (row.lead_score, row.lead_status) == (
                breakdown["lead_score"], breakdown["lead_status"])
            if same and row.reason_source == "template":
                row.reason, row.reason_source = reason, source
        logger.info("lead reason refreshed", extra={"lead_id": lead_id, "written": same})
        return same
    except Exception as exc:
        logger.warning("lead reason refresh failed: %s", type(exc).__name__, extra={"lead_id": lead_id})
        return False


def get_lead(engine: Engine, lead_id: int) -> LeadView | None:
    """One lead with its callbacks, or None."""
    with Session(engine) as session:
        row = session.get(Lead, lead_id)
        return LeadView.model_validate(row) if row else None


def _matches(search: str):
    """Name, matched city or the customer's own city words containing `search`, in any case.

    The text is matched as typed: % and _ (LIKE wildcards) and the escape character are escaped.
    """
    escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    return or_(*(column.ilike(pattern, escape="\\") for column in (Lead.name, Lead.city, Lead.city_input)))


def list_leads(
    engine: Engine,
    status: str | None = None,
    sort: str = "newest",
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
) -> tuple[list[LeadView], int]:
    """(leads, total). Filter by status (HOT, WARM, COLD) and by `search` in the name or city (any
    case); sort by 'newest' or 'score' (highest first). total counts every match, not just this page."""
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}")
    limit, offset = max(1, min(limit, MAX_LIMIT)), max(0, offset)
    filters = []
    if status:
        filters.append(Lead.lead_status == status)
    if search and search.strip():
        filters.append(_matches(search.strip()))
    query = select(Lead).where(*filters)
    count = select(func.count()).select_from(Lead).where(*filters)
    if sort == "score":
        order = (Lead.lead_score.desc(), Lead.created_at.desc(), Lead.id.desc())
    else:
        order = (Lead.created_at.desc(), Lead.id.desc())
    with Session(engine) as session:
        rows = session.scalars(query.order_by(*order).limit(limit).offset(offset)).all()
        return [LeadView.model_validate(r) for r in rows], int(session.scalar(count))


def lead_counts(engine: Engine) -> dict[str, int]:
    """Leads per status in the whole table, e.g. {"HOT": 4, "WARM": 5, "COLD": 3} (one GROUP BY).

    The dashboard's count cards use it, so they stay right however many pages the list has.
    """
    with Session(engine) as session:
        rows = session.execute(select(Lead.lead_status, func.count()).group_by(Lead.lead_status)).all()
    counts = dict.fromkeys(STATUSES, 0)
    counts.update({status: int(n) for status, n in rows if status in counts})
    return counts


def schedule_callback(
    engine: Engine, lead_id: int, preferred_time: str, notes: str | None = None
) -> CallbackResult:
    """Log a callback request for an existing lead (status 'requested'; the team confirms it).

    Raises LeadNotFound without a lead, and InvalidValue for an empty or very long time.
    """
    preferred_time = v.clean_text(preferred_time or "", what="preferred time", max_chars=PREFERRED_TIME_CHARS)
    notes = v.clean_text(notes, what="notes", max_chars=v.LONG_TEXT_CHARS) if notes and notes.strip() else None
    with Session(engine, expire_on_commit=False) as session, session.begin():
        row = session.get(Lead, lead_id)
        if row is None:
            raise LeadNotFound(lead_id)
        callback = Callback(lead_id=row.id, preferred_time=preferred_time, notes=notes, status="requested")
        session.add(callback)
        session.flush()
        session.refresh(callback)
        row.recommended_action = recommended_action(row.score_breakdown, _lead_data(row), preferred_time)
        view = CallbackView.model_validate(callback)
    logger.info("callback requested", extra={"lead_id": lead_id, "callback_id": view.id})
    return CallbackResult(callback=view, message=CALLBACK_MESSAGE)


def create_support_request(
    engine: Engine, raw: dict[str, Any], *, session_id: str | None = None
) -> SupportRequestResult:
    """Save an existing customer's issue. Needs a name, a phone or email, and the issue. Never a lead."""
    checks = {
        "name": v.clean_name,
        "phone": v.clean_phone,
        "email": v.clean_email,
        "issue": lambda value: v.clean_text(value, what="issue", max_chars=v.LONG_TEXT_CHARS),
        "project_details": lambda value: v.clean_text(
            value, what="project details", max_chars=v.LONG_TEXT_CHARS
        ),
    }
    clean: dict[str, str] = {}
    errors: dict[str, str] = {}
    for key, value in raw.items():
        if key not in checks:
            errors[key] = "Unknown field."
        elif value is not None and str(value).strip():
            try:
                clean[key] = checks[key](value)
            except v.InvalidValue as exc:
                errors[key] = str(exc)
    missing = []
    if "name" not in clean:
        missing.append("name")
    if not ("phone" in clean or "email" in clean):
        missing.append("phone or email")
    if "issue" not in clean:
        missing.append("issue")
    if missing:
        return SupportRequestResult(missing_fields=missing, errors=errors)
    with Session(engine) as session, session.begin():
        request = SupportRequest(session_id=session_id, status="open", **clean)
        session.add(request)
        session.flush()
        request_id = request.id
    logger.info("support request created", extra={"support_request_id": request_id})
    return SupportRequestResult(id=request_id, errors=errors)


def list_support_requests(
    engine: Engine, limit: int = 50, offset: int = 0
) -> tuple[list[SupportRequestView], int]:
    """(support requests, total), newest first."""
    limit, offset = max(1, min(limit, MAX_LIMIT)), max(0, offset)
    query = select(SupportRequest).order_by(SupportRequest.created_at.desc(), SupportRequest.id.desc())
    with Session(engine) as session:
        rows = session.scalars(query.limit(limit).offset(offset)).all()
        total = session.scalar(select(func.count()).select_from(SupportRequest))
        return [SupportRequestView.model_validate(r) for r in rows], int(total)

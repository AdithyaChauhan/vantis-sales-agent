"""Sales API (header X-API-Key): leads, callbacks and support requests for the dashboard.

Every route calls the T7 service functions (app/leads/service.py), so the API applies the same checks
and scoring as the agent. session_id is never returned: it is the key to the customer's chat.
Errors are JSON: 401/429 (key), 404 (unknown id), 422 (field messages), 503 (database down).
"""
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Body, Depends, Query
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import Engine

from app.api.security import ApiError, require_sales_key
from app.core.db import get_engine
from app.leads import service
from app.leads import validators as v
from app.leads.schemas import CallbackView, LeadData, LeadView, SupportRequestView
from app.services.service_areas import ServiceAreas

router = APIRouter(prefix="/api", tags=["sales"], dependencies=[Depends(require_sales_key)])

DB_NOT_CONFIGURED = "The database is not configured on this server."
RAHUL_EXAMPLE = {  # the demo lead (PLAN.md part 2); executing it in /docs saves a real lead
    "name": "Rahul Sharma",
    "phone": "9876543210",
    "email": "rahul.sharma@example.com",
    "city": "Lucknow",
    "property_type": "house",
    "system_size_kw": 5,
    "monthly_bill_inr": 4000,
    "timeline": "next month",
    "timeline_bucket": "lt_1m",
    "intent_level": "ready",
}


def leads_engine(engine: Engine | None = Depends(get_engine)) -> Engine:
    """The database engine, or 503 when DATABASE_URL is not set."""
    if engine is None:
        raise ApiError(503, DB_NOT_CONFIGURED)
    return engine


def get_reason_llm() -> Runnable | None:
    """The model that rewrites a new lead's reason after the response. None: the service uses fast_llm().
    Tests override this with a fake."""
    return None


def get_lead_areas() -> ServiceAreas | None:
    """The KB Section 10 list for the city check. None: the service loads it (location 'unknown' if it
    can't). Tests override this with a small fixed list."""
    return None


class LeadListItem(BaseModel):
    """One row of the dashboard's lead table."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str | None
    lead_score: int
    lead_status: str
    city: str | None
    location_status: str
    requirement: str | None
    system_size_kw: float | None
    monthly_bill_inr: int | None
    recommended_action: str
    created_at: datetime


class LeadList(BaseModel):
    items: list[LeadListItem]
    total: int
    limit: int
    offset: int


class LeadDetail(LeadView):
    """A lead with every field, the score breakdown, reason, recommended action and callbacks."""

    session_id: SkipJsonSchema[str | None] = Field(default=None, exclude=True)


class CallbackRequest(BaseModel):
    """A callback request for an existing lead. It is logged as 'requested'; the team confirms it."""

    model_config = ConfigDict(extra="forbid")

    lead_id: int = Field(gt=0)
    preferred_time: str = Field(description="When the customer wants a call, in their words",
                                examples=["tomorrow evening"])
    notes: str | None = None

    @field_validator("preferred_time")
    @classmethod
    def _time(cls, value: str) -> str:
        return v.clean_text(value, what="preferred time", max_chars=service.PREFERRED_TIME_CHARS)

    @field_validator("notes")
    @classmethod
    def _notes(cls, value: str | None) -> str | None:
        return v.clean_text(value, what="notes", max_chars=v.LONG_TEXT_CHARS) if value and value.strip() else None


class CallbackCreated(CallbackView):
    message: str


class SupportRequestList(BaseModel):
    items: list[SupportRequestView]
    total: int
    limit: int
    offset: int


@router.get("/leads", response_model=LeadList)
def list_leads(
    status: Literal["HOT", "WARM", "COLD"] | None = None,
    sort: Literal["newest", "score"] = "newest",
    limit: int = Query(50, ge=1, le=service.MAX_LIMIT),
    offset: int = Query(0, ge=0),
    engine: Engine = Depends(leads_engine),
):
    """Leads for the dashboard, newest first (or highest score first), optionally one status."""
    items, total = service.list_leads(engine, status=status, sort=sort, limit=limit, offset=offset)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/leads/{lead_id}", response_model=LeadDetail)
def get_lead(lead_id: int, engine: Engine = Depends(leads_engine)):
    """One lead with all fields, the score breakdown, reason, recommended action and callbacks."""
    lead = service.get_lead(engine, lead_id)
    if lead is None:
        raise ApiError(404, f"No lead with id {lead_id}.")
    return lead


@router.post("/leads", status_code=201, response_model=LeadDetail)
def create_lead(
    background: BackgroundTasks,
    lead: LeadData = Body(openapi_examples={"demo lead": {"value": RAHUL_EXAMPLE}}),
    engine: Engine = Depends(leads_engine),
    reason_llm: Runnable | None = Depends(get_reason_llm),
    areas: ServiceAreas | None = Depends(get_lead_areas),
):
    """Create a lead by hand: validated and scored exactly like a chat lead.

    422 lists invalid fields, or the missing ones (name, phone or email, city, and a need); nothing is
    saved then. The reason starts as the template sentence; the LLM rewrites it after the response.
    """
    result = service.create_lead(engine, lead.model_dump(exclude_none=True), areas=areas, reason_now=False)
    if result.lead is None:
        raise ApiError(422, "The lead is missing required details.", missing_fields=result.missing_fields)
    background.add_task(service.refresh_reason, engine, result.lead.id, reason_llm)
    return result.lead


@router.post("/callback", status_code=201, response_model=CallbackCreated)
def schedule_callback(request: CallbackRequest, engine: Engine = Depends(leads_engine)):
    """Log a callback request for a lead (status 'requested', never booked). 404 if the lead doesn't exist."""
    try:
        result = service.schedule_callback(engine, request.lead_id, request.preferred_time, request.notes)
    except service.LeadNotFound:
        raise ApiError(404, f"No lead with id {request.lead_id}.") from None
    return {**result.callback.model_dump(), "message": result.message}


@router.get("/support-requests", response_model=SupportRequestList)
def list_support_requests(
    limit: int = Query(50, ge=1, le=service.MAX_LIMIT),
    offset: int = Query(0, ge=0),
    engine: Engine = Depends(leads_engine),
):
    """Existing customers' support requests, newest first."""
    items, total = service.list_support_requests(engine, limit=limit, offset=offset)
    return {"items": items, "total": total, "limit": limit, "offset": offset}

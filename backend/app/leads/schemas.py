"""Lead data contract for T8 (structured output of the understand step) and T9 (POST /leads).

Every field is optional; null means unknown. Values are always clean (see validators.py), units are in
the field names (₹ per month, kW, kWh per month).
"""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.leads import validators as v

PropertyType = Literal["house", "villa", "apartment", "shop", "office", "clinic", "warehouse", "other"]
CustomerType = Literal["residential", "commercial"]
Ownership = Literal["own", "rent"]
TimelineBucket = Literal["lt_1m", "1_3m", "3_6m", "gt_6m"]
IntentLevel = Literal["ready", "interested", "exploring"]
LeadStatus = Literal["HOT", "WARM", "COLD"]
LocationStatus = Literal["listed", "unverified", "unknown"]
Channel = Literal["web", "whatsapp"]

# KB Section 2: homes are residential; shops, offices, clinics and warehouses are small commercial.
CUSTOMER_TYPE_OF = {
    "house": "residential", "villa": "residential", "apartment": "residential",
    "shop": "commercial", "office": "commercial", "clinic": "commercial", "warehouse": "commercial",
}

_TEXT_LIMITS = {
    "city": v.SHORT_TEXT_CHARS,
    "state": v.SHORT_TEXT_CHARS,
    "business_type": v.SHORT_TEXT_CHARS,
    "requirement": v.TEXT_CHARS,
    "timeline": v.TEXT_CHARS,
    "motivation": v.TEXT_CHARS,
    "other_info": v.LONG_TEXT_CHARS,
}
_BOOL_FIELDS = ("rooftop_available", "existing_solar", "requirement_outside_catalog")


class LeadData(BaseModel):
    """What a customer told us about themselves and their solar need."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, description="Customer's name")
    phone: str | None = Field(None, description="Indian mobile number; stored as +91 and 10 digits")
    email: str | None = Field(None, description="Email address")
    city: str | None = Field(None, description="City where solar would be installed")
    state: str | None = Field(None, description="Indian state")
    customer_type: CustomerType | None = Field(
        None, description="residential or commercial; derived from property_type when not given"
    )
    property_type: PropertyType | None = Field(None, description="Kind of property; 'other' if none fits")
    ownership: Ownership | None = Field(None, description="own or rent")
    rooftop_available: bool | None = Field(None, description="Whether a usable rooftop is available")
    monthly_bill_inr: int | None = Field(None, description="Approximate electricity bill, ₹ per month")
    monthly_units_kwh: int | None = Field(None, description="Approximate electricity use, kWh per month")
    system_size_kw: float | None = Field(None, description="Desired system size in kW, if the customer said")
    existing_solar: bool | None = Field(None, description="Whether the property already has solar")
    requirement: str | None = Field(None, description="What the customer wants, in their words")
    timeline: str | None = Field(None, description="When they want it installed, in their words")
    timeline_bucket: TimelineBucket | None = Field(
        None,
        description="lt_1m = now, this month or next month; 1_3m = in 1-3 months; "
        "3_6m = in 3-6 months; gt_6m = later than 6 months",
    )
    intent_level: IntentLevel | None = Field(
        None,
        description="ready = asks for a quote, installation, site visit or callback, or plans to install "
        "soon; interested = wants solar and asks about options; exploring = just curious or far-off plans",
    )
    requirement_outside_catalog: bool | None = Field(
        None, description="True if the need clearly falls outside the KB's reference packages"
    )
    motivation: str | None = Field(None, description="Why the customer is considering solar")
    budget_inr: int | None = Field(None, description="Budget in ₹, if the customer shared it")
    business_type: str | None = Field(None, description="Type of business, for commercial customers")
    other_info: str | None = Field(None, description="Any other relevant detail (never forced into an enum)")

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, value):
        return None if _blank(value) else v.clean_name(value)

    @field_validator("phone", mode="before")
    @classmethod
    def _phone(cls, value):
        return None if _blank(value) else v.clean_phone(value)

    @field_validator("email", mode="before")
    @classmethod
    def _email(cls, value):
        return None if _blank(value) else v.clean_email(value)

    @field_validator("monthly_bill_inr", mode="before")
    @classmethod
    def _bill(cls, value):
        return None if _blank(value) else v.parse_bill(value)

    @field_validator("budget_inr", mode="before")
    @classmethod
    def _budget(cls, value):
        return None if _blank(value) else v.parse_budget(value)

    @field_validator("system_size_kw", mode="before")
    @classmethod
    def _size(cls, value):
        return None if _blank(value) else v.parse_kw(value)

    @field_validator("monthly_units_kwh", mode="before")
    @classmethod
    def _units(cls, value):
        return None if _blank(value) else v.parse_kwh(value)

    @field_validator(*_TEXT_LIMITS, mode="before")
    @classmethod
    def _text(cls, value, info):
        what = info.field_name.replace("_", " ")
        return None if _blank(value) else v.clean_text(value, what=what, max_chars=_TEXT_LIMITS[info.field_name])

    @field_validator(*_BOOL_FIELDS, mode="before")
    @classmethod
    def _bool(cls, value, info):
        return None if _blank(value) else v.parse_bool(value, what=info.field_name.replace("_", " "))

    @field_validator(
        "customer_type", "property_type", "ownership", "timeline_bucket", "intent_level", mode="before"
    )
    @classmethod
    def _enum(cls, value):
        return None if _blank(value) else str(value).strip().lower()

    @model_validator(mode="after")
    def _derive_customer_type(self) -> "LeadData":
        if self.customer_type is None and self.property_type in CUSTOMER_TYPE_OF:
            self.customer_type = CUSTOMER_TYPE_OF[self.property_type]
        return self

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> tuple["LeadData", dict[str, str]]:
        """Validate field by field: keep the valid values, return a message for each invalid one.

        Unlike LeadData(**raw), one bad value (say a 9-digit phone) does not throw away the rest.
        """
        clean: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for key, value in raw.items():
            if key not in cls.model_fields:
                errors[key] = "Unknown field."
                continue
            try:
                clean[key] = getattr(cls.model_validate({key: value}), key)
            except ValidationError as exc:
                errors[key] = error_message(exc)
        return cls.model_validate(clean), errors


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def error_message(exc: ValidationError) -> str:
    """The first error as a plain sentence (our own messages, without Pydantic's 'Value error, ')."""
    return error_text(exc.errors()[0])


def error_text(error: dict) -> str:
    """One Pydantic error (an item of exc.errors()) as a plain sentence. Also used for the API's 422s."""
    if error["type"] == "literal_error":
        return f"Please choose one of: {error['ctx']['expected']}."
    if error["type"] == "extra_forbidden":
        return "Unknown field."
    return str(error.get("ctx", {}).get("error") or error["msg"])


class CallbackView(BaseModel):
    """A callback request as stored."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    lead_id: int
    preferred_time: str
    notes: str | None
    status: str
    created_at: datetime


class LeadView(LeadData):
    """A stored lead with its score, status, reason and recommended action (T9 returns this)."""

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: int
    city_input: str | None
    location_status: LocationStatus
    lead_score: int
    lead_status: LeadStatus
    score_breakdown: dict
    reason: str
    reason_source: Literal["llm", "template"]
    recommended_action: str
    source_channel: Channel
    session_id: str | None
    created_at: datetime
    updated_at: datetime
    callbacks: list[CallbackView] = []


class CreateLeadResult(BaseModel):
    """create_lead's answer: the lead, or what is still missing (nothing is saved then)."""

    lead: LeadView | None = None
    missing_fields: list[str] = []
    errors: dict[str, str] = {}


class UpdateLeadResult(BaseModel):
    """update_lead's answer: the re-scored lead and any values that were rejected."""

    lead: LeadView
    errors: dict[str, str] = {}


class CallbackResult(BaseModel):
    """schedule_callback's answer. The callback is a request; the team confirms it."""

    callback: CallbackView
    message: str


class SupportRequestView(BaseModel):
    """A stored support request, for the sales team (no session_id: it is the key to the customer's chat)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    phone: str | None
    email: str | None
    project_details: str | None
    issue: str
    status: str
    created_at: datetime


class SupportRequestResult(BaseModel):
    """create_support_request's answer: the new id, or what is still missing."""

    id: int | None = None
    missing_fields: list[str] = []
    errors: dict[str, str] = {}

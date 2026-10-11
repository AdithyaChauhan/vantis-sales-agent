"""Tables for leads, callback requests and support requests (created by init_db).

create_all never changes a table that already exists; see PLAN.md part 3.10 for column changes.
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class Lead(Base):
    """A qualified sales lead: what the customer told us, plus the score, status, reason and action."""

    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    source_channel: Mapped[str] = mapped_column(String(20), default="web")

    # LeadData fields (app/leads/schemas.py); city is the matched service city when listed
    name: Mapped[str | None] = mapped_column(String(100))
    phone: Mapped[str | None] = mapped_column(String(13))
    email: Mapped[str | None] = mapped_column(String(254))
    city: Mapped[str | None] = mapped_column(String(100))
    city_input: Mapped[str | None] = mapped_column(String(100))  # the customer's own words
    state: Mapped[str | None] = mapped_column(String(100))
    customer_type: Mapped[str | None] = mapped_column(String(20))
    property_type: Mapped[str | None] = mapped_column(String(20))
    ownership: Mapped[str | None] = mapped_column(String(10))
    rooftop_available: Mapped[bool | None] = mapped_column(Boolean)
    monthly_bill_inr: Mapped[int | None] = mapped_column(Integer)
    monthly_units_kwh: Mapped[int | None] = mapped_column(Integer)
    system_size_kw: Mapped[float | None] = mapped_column(Float)
    existing_solar: Mapped[bool | None] = mapped_column(Boolean)
    requirement: Mapped[str | None] = mapped_column(Text)
    timeline: Mapped[str | None] = mapped_column(Text)
    timeline_bucket: Mapped[str | None] = mapped_column(String(10))
    intent_level: Mapped[str | None] = mapped_column(String(20))
    requirement_outside_catalog: Mapped[bool | None] = mapped_column(Boolean)
    motivation: Mapped[str | None] = mapped_column(Text)
    budget_inr: Mapped[int | None] = mapped_column(Integer)
    business_type: Mapped[str | None] = mapped_column(String(100))
    other_info: Mapped[str | None] = mapped_column(Text)

    # Qualification (app/leads/scoring.py, app/leads/reason.py)
    location_status: Mapped[str] = mapped_column(String(20))
    lead_score: Mapped[int] = mapped_column(Integer)
    lead_status: Mapped[str] = mapped_column(String(10))
    score_breakdown: Mapped[dict] = mapped_column(JSONB)
    reason: Mapped[str] = mapped_column(Text)
    reason_source: Mapped[str] = mapped_column(String(10))
    recommended_action: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    callbacks: Mapped[list["Callback"]] = relationship(
        back_populates="lead", order_by="Callback.id", lazy="selectin", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_leads_lead_status", "lead_status"),
        Index("ix_leads_lead_score", "lead_score"),
    )


class Callback(Base):
    """A callback the customer asked for. Status starts as 'requested'; it is never 'booked' by the AI."""

    __tablename__ = "callbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lead_id: Mapped[int] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    preferred_time: Mapped[str] = mapped_column(Text)  # exactly as the customer said it
    notes: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="requested")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lead: Mapped[Lead] = relationship(back_populates="callbacks")


class SupportRequest(Base):
    """An issue from an existing customer (e.g. an inverter fault). Never a sales lead."""

    __tablename__ = "support_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(100))
    phone: Mapped[str | None] = mapped_column(String(13))
    email: Mapped[str | None] = mapped_column(String(254))
    project_details: Mapped[str | None] = mapped_column(Text)
    issue: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

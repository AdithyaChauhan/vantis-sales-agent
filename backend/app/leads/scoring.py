"""Lead scoring, status, caps, recommended action and the minimum fields for a lead.

All rules are fixed code (never the LLM), so the same input always gives the same result.
Plain-language version: docs/SCORING.md.
"""
from app.leads.schemas import LeadData

HOT_MIN = 75
WARM_MIN = 45
CAPPED_MAX = HOT_MIN - 1  # a capped lead is at most WARM, and its score at most 74

INTENT_POINTS = {"ready": 25, "interested": 15, "exploring": 5, None: 0}
REQUIREMENT_POINTS = {"clear": 20, "vague": 10, "none": 0}
LOCATION_POINTS = {"listed": 20, "unverified": 5, "unknown": 0}
TIMELINE_POINTS = {"lt_1m": 20, "1_3m": 15, "3_6m": 8, "gt_6m": 3, None: 0}
CONTACT_POINTS = {"both": 10, "one": 6, "none": 0}
PROPERTY_POINTS = {"owner_with_rooftop": 5, "partly_known": 2, "renting_or_no_rooftop": 0, "unknown": 0}

INTENT_LABELS = {
    "ready": "Ready to buy", "interested": "Interested in options",
    "exploring": "Only exploring", None: "Intent not clear",
}
LOCATION_LABELS = {
    "listed": "City is in the listed service areas",
    "unverified": "City is not in the listed service areas",
    "unknown": "Location not known",
}
TIMELINE_LABELS = {
    "lt_1m": "Wants it within about a month", "1_3m": "Wants it in 1-3 months",
    "3_6m": "Wants it in 3-6 months", "gt_6m": "Wants it in more than 6 months",
    None: "Timeline not known",
}
CAP_LABELS = {
    "unverified_location": "City not in the listed service areas: capped at WARM",
    "outside_catalog": "Requirement outside the reference packages: capped at WARM",
}

HOT_ACTION = "Contact the lead within 1 hour and offer a site assessment."
WARM_ACTION = "Follow up within 24 hours."
COLD_ACTION = "Add to the nurture list, share product information and follow up in 30 days."


def _rule(rule: str, value, points: int, max_points: int, label: str) -> dict:
    return {"rule": rule, "value": value, "points": points, "max": max_points, "label": label}


def _requirement(lead: LeadData) -> dict:
    """Clear = a size, a bill or a usage figure. Labels never contain money amounts (they go to the LLM)."""
    given = []
    if lead.system_size_kw:
        given.append(f"{lead.system_size_kw:g} kW system")
    if lead.monthly_bill_inr:
        given.append("monthly bill given")
    if lead.monthly_units_kwh:
        given.append("monthly usage given")
    if given:
        value, label = "clear", "Clear requirement: " + ", ".join(given)
    elif lead.requirement:
        value, label = "vague", "Requirement described in words only"
    else:
        value, label = "none", "No requirement yet"
    return _rule("requirement", value, REQUIREMENT_POINTS[value], 20, label)


def _contact(lead: LeadData) -> dict:
    if lead.phone and lead.email:
        value, label = "both", "Phone and email given"
    elif lead.phone or lead.email:
        value, label = "one", "Phone given" if lead.phone else "Email given"
    else:
        value, label = "none", "No contact details"
    return _rule("contact", value, CONTACT_POINTS[value], 10, label)


def _property(lead: LeadData) -> dict:
    """5 = owns it and has a rooftop; 0 = renting or no rooftop; 2 = anything else known; 0 = nothing."""
    if lead.ownership == "own" and lead.rooftop_available:
        value, label = "owner_with_rooftop", "Owner with a usable rooftop"
    elif lead.ownership == "rent" or lead.rooftop_available is False:
        value, label = "renting_or_no_rooftop", "Renting or no usable rooftop"
    elif lead.property_type or lead.ownership or lead.rooftop_available is not None:
        value, label = "partly_known", "Property partly known"
    else:
        value, label = "unknown", "Property not known"
    return _rule("property", value, PROPERTY_POINTS[value], 5, label)


def status_for(score: int) -> str:
    """HOT at 75 or more, WARM at 45-74, COLD below 45."""
    if score >= HOT_MIN:
        return "HOT"
    return "WARM" if score >= WARM_MIN else "COLD"


def score_lead(lead: LeadData, location_status: str) -> dict:
    """The score breakdown: every rule with its points and label, the caps, the score and the status."""
    rules = [
        _rule("intent", lead.intent_level, INTENT_POINTS[lead.intent_level], 25, INTENT_LABELS[lead.intent_level]),
        _requirement(lead),
        _rule(
            "location", location_status, LOCATION_POINTS[location_status], 20, LOCATION_LABELS[location_status]
        ),
        _rule(
            "timeline", lead.timeline_bucket, TIMELINE_POINTS[lead.timeline_bucket], 20,
            TIMELINE_LABELS[lead.timeline_bucket],
        ),
        _contact(lead),
        _property(lead),
    ]
    raw_score = sum(r["points"] for r in rules)
    caps = []
    if location_status == "unverified":
        caps.append({"rule": "unverified_location", "label": CAP_LABELS["unverified_location"]})
    if lead.requirement_outside_catalog:
        caps.append({"rule": "outside_catalog", "label": CAP_LABELS["outside_catalog"]})
    lead_score = min(raw_score, CAPPED_MAX) if caps else raw_score
    return {
        "rules": rules,
        "raw_score": raw_score,
        "caps": caps,
        "lead_score": lead_score,
        "lead_status": status_for(lead_score),
    }


def missing_details(lead: LeadData, breakdown: dict) -> list[str]:
    """Details worth collecting for a WARM lead (each would add points)."""
    rules = {r["rule"]: r for r in breakdown["rules"]}
    missing = []
    if lead.timeline_bucket is None:
        missing.append("timeline")
    if rules["requirement"]["value"] != "clear":
        missing.append("monthly bill or usage")
    if not lead.phone:
        missing.append("phone number")
    if not lead.email:
        missing.append("email")
    if rules["property"]["value"] in ("partly_known", "unknown"):
        if lead.ownership is None:
            missing.append("ownership")
        if lead.rooftop_available is None:
            missing.append("rooftop availability")
    return missing


def recommended_action(breakdown: dict, lead: LeadData, callback_time: str | None = None) -> str:
    """Fixed next step for the sales team, from the status (plus the gaps for a WARM lead)."""
    status = breakdown["lead_status"]
    if status == "HOT":
        parts = [HOT_ACTION]
    elif status == "COLD":
        parts = [COLD_ACTION]
    else:
        parts = [WARM_ACTION]
        caps = {c["rule"] for c in breakdown["caps"]}
        if "unverified_location" in caps:
            parts.append(f"Verify service availability for {lead.city}.")
        if "outside_catalog" in caps:
            parts.append("Check whether the requirement fits the reference packages.")
        if lead.ownership == "rent":
            parts.append("Confirm the owner's permission for a rooftop installation.")
        missing = missing_details(lead, breakdown)
        if missing:
            parts.append("Collect: " + ", ".join(missing) + ".")
    if callback_time:
        parts.append(f"Customer asked for a callback: {callback_time}.")
    return " ".join(parts)


def missing_for_creation(lead: LeadData) -> list[str]:
    """What must still be asked before a lead is saved (empty list = ready to save).

    Name, AND phone or email, AND city, AND some need (requirement, bill, usage, size or property type).
    """
    missing = []
    if not lead.name:
        missing.append("name")
    if not (lead.phone or lead.email):
        missing.append("phone or email")
    if not lead.city:
        missing.append("city")
    has_need = any(
        (lead.requirement, lead.monthly_bill_inr, lead.monthly_units_kwh, lead.system_size_kw, lead.property_type)
    )
    if not has_need:
        missing.append("requirement, monthly bill, usage, system size or property type")
    return missing

"""Lead scoring, status, caps, recommended action and the minimum fields for a lead.

All rules are fixed code (never the LLM), so the same input always gives the same result.
Plain-language version: docs/SCORING.md.
"""
from app.leads.schemas import LeadData

HOT_MIN = 75
WARM_MIN = 45
CAPPED_MAX = HOT_MIN - 1  # a capped lead is at most WARM, and its score at most 74
COLD_MAX = WARM_MIN - 1  # T13d: the exploring cap keeps a lead COLD, and its score at most 44

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
    "exploring": "Only exploring, with no timeline under 3 months: capped at COLD",
}
# The highest score each cap allows (T13d: a cap now has its own limit)
CAP_MAX = {"unverified_location": CAPPED_MAX, "outside_catalog": CAPPED_MAX, "exploring": COLD_MAX}
# A timeline under 3 months: an exploring lead with one is not capped (the assignment's Cold: "exploratory intent")
NEAR_TIMELINES = ("lt_1m", "1_3m")

HOT_ACTION = "Contact the lead within 1 hour and offer a site assessment."
WARM_ACTION = "Follow up within 24 hours."
COLD_ACTION = "Add to the nurture list, share product information and follow up in 30 days."
COLD_AFTER_CALLBACK = "Add to the nurture list and share product information."
# T13d: a COLD lead with ready intent asked to be contacted (or is ready to buy): a call, never "follow up in 30 days"
# (audit S21: "can someone call me?" plus details was saved with "follow up in 30 days")
COLD_READY_ACTION = "Call the customer within 24 hours: they asked to be contacted or are ready to buy."
# T13d (Adithya): the customer asked to be called but gave no time: whatever the status, the action starts with this
CALL_REQUESTED_ACTION = "Call the customer within 24 hours; they asked for a call."

# What keeps a lead from the next status, in a few words for the reason (T12a). Labels only: never a name,
# contact detail or amount. Signals that more details can't change (a city not on the KB list, renting or
# no rooftop) have no label, so they are never named as a gap.
NEXT_STATUS = {"WARM": ("HOT", HOT_MIN), "COLD": ("WARM", WARM_MIN)}
GAP_LABELS = {
    "intent": {None: "intent not clear", "exploring": "only exploring", "interested": "interested but not ready to buy"},
    "requirement": {"none": "no requirement", "vague": "no size, bill or usage"},
    "location": {"unknown": "city not known"},
    "timeline": {None: "no timeline", "gt_6m": "timeline over 6 months", "3_6m": "timeline 3-6 months",
                 "1_3m": "timeline 1-3 months"},
    "contact": {"none": "no contact details"},
    "property": {"unknown": "property not known", "partly_known": "ownership and rooftop not confirmed"},
}
CONTACT_GAPS = {"Phone given": "no email", "Email given": "no phone number"}  # by the contact rule's label
CAP_CAUSES = {
    "unverified_location": "city not in the listed service areas",
    "outside_catalog": "requirement outside the reference packages",
    "exploring": "only exploring, no timeline under 3 months",
}
# The rule a cap's cause belongs to, so a reason never names it twice (T12b); outside_catalog has no rule of its own
CAP_RULES = {"unverified_location": "location", "outside_catalog": "catalog", "exploring": "intent"}
GAP_ORDER = ["timeline", "requirement", "intent", "location", "contact", "property"]  # ties: timeline first


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
    applies = []
    if location_status == "unverified":
        applies.append("unverified_location")
    if lead.requirement_outside_catalog:
        applies.append("outside_catalog")
    # T13d (the assignment's Cold: "exploratory intent"): only exploring stays COLD without a timeline under 3 months.
    # "Just exploring, maybe next year" plus details scored WARM 56 in the audit (S22)
    if lead.intent_level == "exploring" and lead.timeline_bucket not in NEAR_TIMELINES:
        applies.append("exploring")
    # Only the caps that lower the score are recorded (T12a), and of those the strictest (T13d): a lead that the
    # exploring cap keeps at 44 does not also list the 74 caps
    lowering = [rule for rule in applies if raw_score > CAP_MAX[rule]]
    limit = min((CAP_MAX[rule] for rule in lowering), default=None)
    caps = [{"rule": rule, "label": CAP_LABELS[rule]} for rule in lowering if CAP_MAX[rule] == limit]
    lead_score = min(raw_score, limit) if limit is not None else raw_score
    return {
        "rules": rules,
        "raw_score": raw_score,
        "caps": caps,
        "lead_score": lead_score,
        "lead_status": status_for(lead_score),
    }


def missing_details(lead: LeadData, breakdown: dict) -> list[str]:
    """Details that would still add points, most useful first (T12a): the timeline (up to 20), a size, bill or
    usage figure when the need is only in words (+10), rooftop and ownership (+3 to +5 together), then email
    or phone (+4). A detail that is known never appears."""
    rules = {r["rule"]: r for r in breakdown["rules"]}
    missing = []
    if lead.timeline_bucket is None:
        missing.append("timeline")
    if rules["requirement"]["value"] != "clear":
        missing.append("monthly bill or usage")
    if rules["property"]["value"] in ("partly_known", "unknown"):
        if lead.rooftop_available is None:
            missing.append("rooftop availability")
        if lead.ownership is None:
            missing.append("ownership")
    if not lead.email:
        missing.append("email")
    if not lead.phone:
        missing.append("phone number")
    return missing


def _gap_label(rule: dict) -> str | None:
    """The few words for a rule that is short of its maximum, or None when more details can't change it."""
    if rule["points"] >= rule["max"]:
        return None
    if rule["rule"] == "contact" and rule["value"] == "one":
        return CONTACT_GAPS.get(rule["label"])
    return GAP_LABELS.get(rule["rule"], {}).get(rule["value"])


def status_gaps(breakdown: dict) -> tuple[str, list[tuple[str, str]]] | None:
    """What keeps a lead from the next status, with the rule each gap belongs to (T12b):
    ("HOT", [("timeline", "no timeline")]), or None for a HOT lead. A cap's cause has its CAP_RULES rule.

    A capped lead: the cap's cause (without it the lead would be HOT). Otherwise the missing signals worth the
    most points, most first (timeline first on a tie), until they would reach the next status's score.
    """
    status = breakdown["lead_status"]
    if status not in NEXT_STATUS:
        return None
    next_status, threshold = NEXT_STATUS[status]
    if breakdown["caps"]:
        return next_status, [(CAP_RULES[c["rule"]], CAP_CAUSES[c["rule"]]) for c in breakdown["caps"]]
    gaps = []
    for rule in breakdown["rules"]:
        label = _gap_label(rule)
        if label:
            gaps.append((rule["max"] - rule["points"], GAP_ORDER.index(rule["rule"]), rule["rule"], label))
    gaps.sort(key=lambda gap: (-gap[0], gap[1]))
    needed, gained, named = threshold - breakdown["lead_score"], 0, []
    for gain, _, rule, label in gaps:
        if gained >= needed:
            break
        named.append((rule, label))
        gained += gain
    return next_status, named


def next_status_gap(breakdown: dict) -> tuple[str, list[str]] | None:
    """What keeps a lead from the next status (T12a): ("HOT", ["no timeline"]), or None for a HOT lead."""
    gap = status_gaps(breakdown)
    return None if gap is None else (gap[0], [label for _, label in gap[1]])


def recommended_action(breakdown: dict, lead: LeadData, callback_time: str | None = None,
                       call_requested: bool = False) -> str:
    """Fixed next step for the sales team, from the status (plus the gaps for a WARM lead).

    A callback time the customer asked for replaces the status's own timing (T12a), so the action never says
    "within 24 hours" next to "the day after tomorrow". T13d: a call they asked for without a time starts the action
    with CALL_REQUESTED_ACTION, whatever the status.
    """
    status = breakdown["lead_status"]
    call = f"Call {callback_time.strip().rstrip('.!?')}, as requested" if callback_time else None
    asked = call is None and call_requested
    if status == "HOT":
        if asked:
            return f"{CALL_REQUESTED_ACTION} Offer a site assessment."
        return f"{call}, and offer a site assessment." if call else HOT_ACTION
    if status == "COLD":
        if call:
            return f"{call}. {COLD_AFTER_CALLBACK}"
        if asked:
            return CALL_REQUESTED_ACTION
        return COLD_READY_ACTION if lead.intent_level == "ready" else COLD_ACTION
    parts = [f"{call}." if call else CALL_REQUESTED_ACTION if asked else WARM_ACTION]
    rules = {r["rule"]: r for r in breakdown["rules"]}
    if rules["location"]["value"] == "unverified":
        parts.append(f"Verify service availability for {lead.city}.")
    if lead.requirement_outside_catalog:
        parts.append("Check whether the requirement fits the reference packages.")
    if lead.ownership == "rent":
        parts.append("Confirm the owner's permission for a rooftop installation.")
    missing = missing_details(lead, breakdown)
    if missing:
        parts.append("Collect: " + ", ".join(missing) + ".")
    return " ".join(parts)


def missing_for_creation(lead: LeadData) -> list[str]:
    """What must still be asked before a lead is saved (empty list = ready to save).

    Name, AND phone or email, AND city, AND some need (requirement, bill, usage, size, property type, or T13d: whether
    it is a home or a business; audit S19: "for my home" with name, phone and city saved nothing).
    """
    missing = []
    if not lead.name:
        missing.append("name")
    if not (lead.phone or lead.email):
        missing.append("phone or email")
    if not lead.city:
        missing.append("city")
    has_need = any((lead.requirement, lead.monthly_bill_inr, lead.monthly_units_kwh, lead.system_size_kw,
                    lead.property_type, lead.customer_type))
    if not has_need:
        missing.append("requirement, monthly bill, usage, system size or property type")
    return missing

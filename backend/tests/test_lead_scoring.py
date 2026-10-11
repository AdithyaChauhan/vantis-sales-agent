"""Scoring rules, status thresholds, caps, recommended actions and the minimum fields for a lead."""
import pytest

from app.leads.schemas import LeadData
from app.leads.scoring import (
    COLD_ACTION,
    HOT_ACTION,
    WARM_ACTION,
    missing_for_creation,
    recommended_action,
    score_lead,
    status_for,
)
from app.leads.service import locate
from tests.leads_data import AREAS, EXPLORER, RAHUL


def scored(raw: dict, location_status: str | None = None) -> tuple[LeadData, dict]:
    data, errors = LeadData.from_raw(raw)
    assert errors == {}
    status = location_status or locate(data.city, data.state, AREAS).status
    return data, score_lead(data, status)


def points(breakdown: dict) -> dict[str, int]:
    return {r["rule"]: r["points"] for r in breakdown["rules"]}


def test_rahul_in_lucknow_is_hot():
    _, b = scored(RAHUL)
    assert points(b) == {"intent": 25, "requirement": 20, "location": 20, "timeline": 20, "contact": 10, "property": 2}
    assert (b["raw_score"], b["lead_score"], b["lead_status"], b["caps"]) == (97, 97, "HOT", [])


def test_rahul_in_pune_is_capped_at_warm():
    _, b = scored({**RAHUL, "city": "Pune"})
    assert points(b)["location"] == 5
    assert (b["raw_score"], b["lead_score"], b["lead_status"]) == (82, 74, "WARM")
    assert [c["rule"] for c in b["caps"]] == ["unverified_location"]


def test_requirement_outside_the_catalog_is_capped_at_warm():
    _, b = scored({**RAHUL, "requirement_outside_catalog": True})
    assert (b["raw_score"], b["lead_score"], b["lead_status"]) == (97, 74, "WARM")
    assert [c["rule"] for c in b["caps"]] == ["outside_catalog"]


def test_both_caps_together():
    _, b = scored({**RAHUL, "city": "Pune", "requirement_outside_catalog": True})
    assert (b["lead_score"], b["lead_status"]) == (74, "WARM")
    assert len(b["caps"]) == 2


def test_a_cap_never_raises_a_low_score():
    _, b = scored({**EXPLORER, "city": "Pune"})
    assert b["lead_score"] == b["raw_score"] < 45
    assert b["lead_status"] == "COLD"


def test_explorer_is_cold():
    _, b = scored(EXPLORER)
    assert points(b) == {"intent": 5, "requirement": 0, "location": 20, "timeline": 3, "contact": 6, "property": 2}
    assert (b["lead_score"], b["lead_status"]) == (36, "COLD")


@pytest.mark.parametrize(
    "fields, location, total, status",
    [
        # 25 + 20 + 20 + 0 + 10 + 0
        ({"intent_level": "ready", "system_size_kw": 5, "phone": "9876543210", "email": "a@b.co"}, "listed", 75, "HOT"),
        # 25 + 20 + 20 + 3 + 6 + 0
        ({"intent_level": "ready", "system_size_kw": 5, "timeline_bucket": "gt_6m", "phone": "9876543210"}, "listed", 74, "WARM"),
        # 15 + 10 + 20 + 0 + 0 + 0
        ({"intent_level": "interested", "requirement": "solar for my shop"}, "listed", 45, "WARM"),
        # 15 + 10 + 5 + 8 + 6 + 0
        ({"intent_level": "interested", "requirement": "solar", "timeline_bucket": "3_6m", "email": "a@b.co"}, "unverified", 44, "COLD"),
    ],
)
def test_status_boundaries(fields, location, total, status):
    _, b = scored(fields, location)
    assert (b["lead_score"], b["lead_status"]) == (total, status)


def test_thresholds():
    assert [status_for(s) for s in (0, 44, 45, 74, 75, 100)] == ["COLD", "COLD", "WARM", "WARM", "HOT", "HOT"]


@pytest.mark.parametrize(
    "fields, rule, expected",
    [
        ({"intent_level": "ready"}, "intent", 25),
        ({"intent_level": "interested"}, "intent", 15),
        ({"intent_level": "exploring"}, "intent", 5),
        ({}, "intent", 0),
        ({"monthly_bill_inr": 4000}, "requirement", 20),
        ({"monthly_units_kwh": 400}, "requirement", 20),
        ({"system_size_kw": 3}, "requirement", 20),
        ({"requirement": "rooftop solar"}, "requirement", 10),
        ({}, "requirement", 0),
        ({"timeline_bucket": "lt_1m"}, "timeline", 20),
        ({"timeline_bucket": "1_3m"}, "timeline", 15),
        ({"timeline_bucket": "3_6m"}, "timeline", 8),
        ({"timeline_bucket": "gt_6m"}, "timeline", 3),
        ({}, "timeline", 0),
        ({"phone": "9876543210", "email": "a@b.co"}, "contact", 10),
        ({"phone": "9876543210"}, "contact", 6),
        ({"email": "a@b.co"}, "contact", 6),
        ({}, "contact", 0),
        ({"ownership": "own", "rooftop_available": True}, "property", 5),
        ({"ownership": "own"}, "property", 2),
        ({"property_type": "house"}, "property", 2),
        ({"ownership": "rent", "rooftop_available": True}, "property", 0),
        ({"ownership": "own", "rooftop_available": False}, "property", 0),
        ({}, "property", 0),
    ],
)
def test_every_rubric_row(fields, rule, expected):
    _, b = scored(fields, "unknown")
    assert points(b)[rule] == expected


@pytest.mark.parametrize("location, expected", [("listed", 20), ("unverified", 5), ("unknown", 0)])
def test_location_points(location, expected):
    _, b = scored({}, location)
    assert points(b)["location"] == expected


def test_same_input_same_breakdown():
    assert scored(RAHUL)[1] == scored(dict(RAHUL))[1]


def test_breakdown_labels_hold_no_money_or_personal_details():
    _, b = scored(RAHUL)
    text = str(b)
    for secret in ("4000", "4,000", "₹", "Rahul", "9876543210", "@"):
        assert secret not in text
    assert "monthly bill given" in text and "5 kW system" in text


def test_hot_action_follows_the_assignment():
    data, b = scored(RAHUL)
    assert recommended_action(b, data) == HOT_ACTION == "Contact the lead within 1 hour and offer a site assessment."


def test_cold_action():
    data, b = scored(EXPLORER)
    assert recommended_action(b, data) == COLD_ACTION


def test_warm_action_names_the_gaps():
    data, b = scored({**RAHUL, "city": "Pune"})
    action = recommended_action(b, data)
    assert action.startswith(WARM_ACTION)
    assert "Verify service availability for Pune." in action
    assert "Collect: ownership, rooftop availability." in action


def test_warm_action_for_a_renter_outside_the_catalog():
    data, b = scored(
        {**RAHUL, "ownership": "rent", "requirement_outside_catalog": True, "email": None, "timeline_bucket": None}
    )
    action = recommended_action(b, data)
    assert "Check whether the requirement fits the reference packages." in action
    assert "Confirm the owner's permission for a rooftop installation." in action
    assert "Collect: timeline, email." in action


def test_callback_time_is_added_to_any_action():
    data, b = scored(RAHUL)
    action = recommended_action(b, data, callback_time="tomorrow evening")
    assert action == HOT_ACTION + " Customer asked for a callback: tomorrow evening."


def test_minimum_fields_for_a_lead():
    full, _ = LeadData.from_raw(RAHUL)
    assert missing_for_creation(full) == []
    assert missing_for_creation(LeadData()) == [
        "name",
        "phone or email",
        "city",
        "requirement, monthly bill, usage, system size or property type",
    ]
    only_property = LeadData(name="Asha Rao", email="asha@example.com", city="Delhi", property_type="house")
    assert missing_for_creation(only_property) == []
    assert missing_for_creation(LeadData(name="Asha Rao", city="Delhi", property_type="house")) == ["phone or email"]

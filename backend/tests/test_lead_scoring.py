"""Scoring rules, status thresholds, caps, recommended actions and the minimum fields for a lead."""
import pytest

from app.leads.schemas import LeadData
from app.leads.scoring import (
    COLD_ACTION,
    HOT_ACTION,
    WARM_ACTION,
    missing_details,
    missing_for_creation,
    next_status_gap,
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


def test_a_cap_that_changes_nothing_is_not_recorded():
    # T12a: a COLD lead in an unlisted city (the seed's Rohit Patel, 31) carried "capped at WARM" for nothing
    _, cold = scored({**EXPLORER, "city": "Pune"})
    assert cold["caps"] == []
    # A WARM lead under 74 in an unlisted city: no cap, but the action still asks to verify the city
    data, warm = scored({**RAHUL, "city": "Pune", "timeline_bucket": None, "email": None})
    assert (warm["raw_score"], warm["lead_status"], warm["caps"]) == (58, "WARM", [])
    assert "Verify service availability for Pune." in recommended_action(warm, data)
    # Outside the catalog under 74: no cap either, and the action still asks to check the requirement
    data, outside = scored({**RAHUL, "requirement_outside_catalog": True, "timeline_bucket": None, "email": None})
    assert (outside["raw_score"], outside["lead_status"], outside["caps"]) == (73, "WARM", [])
    assert "Check whether the requirement fits the reference packages." in recommended_action(outside, data)


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
    assert "Collect: rooftop availability, ownership." in action


def test_missing_details_most_useful_first_and_never_a_known_one():
    # T12a order: timeline (up to 20), a size/bill/usage figure (+10), rooftop and ownership, then email or phone
    words_only = {"name": "Asha Rao", "phone": "9876543210", "city": "Noida", "requirement": "solar for my office",
                  "property_type": "office"}
    data, b = scored(words_only, "listed")
    assert missing_details(data, b) == ["timeline", "monthly bill or usage", "rooftop availability", "ownership",
                                        "email"]
    data, b = scored({**words_only, "monthly_bill_inr": "2000", "rooftop_available": True}, "listed")
    assert missing_details(data, b) == ["timeline", "ownership", "email"]  # the bill and the roof were given
    data, b = scored({**RAHUL, "ownership": "own", "rooftop_available": True})
    assert missing_details(data, b) == []


@pytest.mark.parametrize("raw, location, expected", [
    # The replay's lead (T12a): Noida, a ₹2,000 bill, words only for the need, phone only, no timeline
    ({"name": "Asha Rao", "phone": "9876543210", "city": "Noida", "monthly_bill_inr": "2000",
      "requirement": "solar for my office", "property_type": "office", "intent_level": "interested"},
     "listed", ("HOT", ["no timeline"])),
    ({**RAHUL, "city": "Pune"}, None, ("HOT", ["city not in the listed service areas"])),  # capped: the cap's cause
    # COLD 36, 9 short: no requirement and only exploring are worth 20 each; on a tie the requirement goes first
    (EXPLORER, None, ("WARM", ["no requirement"])),
    # COLD 21 in an unlisted city, 24 short: two signals; the city is not named (more details can't change it)
    ({**EXPLORER, "city": "Pune"}, None, ("WARM", ["no requirement", "only exploring"])),
    (RAHUL, None, None),  # HOT: nothing above it
])
def test_next_status_gap(raw, location, expected):
    _, b = scored(raw, location)
    assert next_status_gap(b) == expected


def test_next_status_gap_names_more_than_one_signal_when_one_is_not_enough():
    _, b = scored({"name": "Asha Rao", "phone": "9876543210", "city": "Kanpur", "property_type": "house",
                   "intent_level": "exploring"})  # 5 + 0 + 20 + 0 + 6 + 2 = 33, COLD; 12 short of WARM
    assert b["lead_score"] == 33
    assert next_status_gap(b) == ("WARM", ["no timeline"])  # +20 is enough on its own; the timeline goes first
    _, warm = scored({"name": "Asha Rao", "phone": "9876543210", "city": "Kanpur", "property_type": "house",
                      "intent_level": "interested", "requirement": "solar for the house"})  # 15+10+20+0+6+2 = 53
    assert next_status_gap(warm) == ("HOT", ["no timeline", "no size, bill or usage"])  # 22 short: 20 + 10


def test_warm_action_for_a_renter_outside_the_catalog():
    data, b = scored(
        {**RAHUL, "ownership": "rent", "requirement_outside_catalog": True, "email": None, "timeline_bucket": None}
    )
    action = recommended_action(b, data)
    assert "Check whether the requirement fits the reference packages." in action
    assert "Confirm the owner's permission for a rooftop installation." in action
    assert "Collect: timeline, email." in action


def test_a_callback_time_replaces_the_status_timing_in_every_action():
    # T12a: "Follow up within 24 hours" clashed with "callback day after tomorrow"
    data, hot = scored(RAHUL)
    assert recommended_action(hot, data, callback_time="tomorrow evening") == (
        "Call tomorrow evening, as requested, and offer a site assessment.")
    data, warm = scored({**RAHUL, "city": "Pune"})
    action = recommended_action(warm, data, callback_time="day after tomorrow, 4pm")
    assert action.startswith("Call day after tomorrow, 4pm, as requested. Verify service availability for Pune.")
    data, cold = scored(EXPLORER)
    assert recommended_action(cold, data, callback_time="Friday.") == (
        "Call Friday, as requested. Add to the nurture list and share product information.")
    for action in (recommended_action(hot, data, "x"), recommended_action(warm, data, "x"),
                   recommended_action(cold, data, "x")):
        assert "within" not in action and "30 days" not in action


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

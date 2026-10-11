"""Shared test data for the lead tests (synthetic; not the KB)."""
from app.services.service_areas import ServiceAreas

# The demo lead from PLAN.md part 2 (demo script steps 2 and 3)
RAHUL = {
    "name": "Rahul Sharma",
    "phone": "9876543210",
    "email": "rahul.sharma@example.com",
    "city": "Lucknow",
    "property_type": "house",
    "system_size_kw": "5 kW",
    "monthly_bill_inr": "₹4,000",
    "timeline": "next month",
    "timeline_bucket": "lt_1m",
    "intent_level": "ready",
}

# Someone who is only looking around: enough for a lead, few signals
EXPLORER = {
    "name": "Priya Verma",
    "phone": "+91 91234 56789",
    "city": "Kanpur",
    "property_type": "apartment",
    "timeline_bucket": "gt_6m",
    "intent_level": "exploring",
}

AREAS = ServiceAreas.from_rows(
    [
        {"Region": "Uttar Pradesh", "Reference Locations": "Lucknow, Kanpur"},
        {"Region": "Delhi NCR", "Reference Locations": "Delhi, Gurugram"},
    ]
)

"""What code reads from the customer's words when the understand step fails (T13d), and the what-if and go-ahead
checks. No network, no database: a small service-area list."""
import pytest

from app.agent.text_details import (
    asks_to_be_called, asks_what_is_offered, details_from_text, is_go_ahead,
)
from app.leads.schemas import LeadData
from app.services.service_areas import ServiceAreas

AREAS = ServiceAreas.from_rows([
    {"Region": "Uttar Pradesh", "Reference Locations": "Lucknow, Kanpur, Noida"},
    {"Region": "Delhi NCR", "Reference Locations": "Delhi, Gurugram, Noida"},
    {"Region": "Haryana", "Reference Locations": "Karnal"},
])


@pytest.mark.parametrize("message, expected", [
    # Script A, message 9 (Render, 13 Oct 12:03:32 UTC): every model failed and nothing was stored
    ("main Rakesh Sharma hu, mera number 9876543210 hai, email rakesh.sharma@example.com",
     {"name": "Rakesh Sharma", "phone": "9876543210", "email": "rakesh.sharma@example.com"}),
    ("mera bijli bill 3500 aata hai monthly, 3bhk hai, kaunsa system lena chahiye", {"monthly_bill_inr": "3500"}),
    ("My name is Arjun Mehta and I live in Pune. Bill is about ₹4,000. Want a 5 kW system. arjun@example.com "
     "98765 43210", {"name": "Arjun Mehta", "city": "Pune", "monthly_bill_inr": "4,000", "system_size_kw": "5 kW",
                     "email": "arjun@example.com", "phone": "98765 43210"}),
    ("Priya Verma, +91 91234 56789", {"name": "Priya Verma", "phone": "+91 91234 56789"}),
    ("Rohan Verma, 9876500002", {"name": "Rohan Verma", "phone": "9876500002"}),
    ("mera naam Sunita hai, Kanpur mein rehti hu", {"name": "Sunita", "city": "Kanpur"}),
    ("4000 ka bill aata hai", {"monthly_bill_inr": "4000"}),
    ("bill ~4k per month", {"monthly_bill_inr": "4k"}),
    ("I want a 5kw system for my house", {"system_size_kw": "5kw"}),
    ("aap log noida me kaam karte ho", {"city": "Noida"}),
    ("I live in Dehli", {"city": "Delhi"}),  # one swapped pair of letters
    ("the house is in Greater Noida", {"city": "Greater Noida"}),  # a separate place, never Noida
    ("Lucknow ke paas ek gaon mein ghar hai", {"city": "Lucknow ke paas"}),  # a landmark: unverified
    ("Sector 62, Noida", {"city": "Noida"}),  # a locality of a listed city is that city
])
def test_details_from_text(message, expected):
    assert details_from_text(message, AREAS) == expected


@pytest.mark.parametrize("message", [
    "call me on 9876543210",  # contact words are never a name
    "main ghar par hoon",  # "I'm at home": not a name, not a place
    "main Lucknow se hu",  # a city, not a name
    "I'm in a hurry, what's the price?",
    "Karna",  # never fuzzy: not Karnal
    "subsidy milti hai kya sarkar se",
    "Is a 100000 rupee budget enough?",  # an amount, not a phone number
])
def test_no_false_name_city_or_phone(message):
    found = details_from_text(message, AREAS)
    assert "name" not in found
    assert found.get("city") in (None, "Lucknow")
    assert "phone" not in found or found["phone"].replace(" ", "").isdigit() and len(found["phone"]) >= 8


def test_main_lucknow_se_hu_gives_the_city_only():
    assert details_from_text("main Lucknow se hu", AREAS) == {"city": "Lucknow"}


def test_bad_values_reach_the_validators_and_become_questions():
    """S18: an invalid email (or a short phone) is read, so the reply says what looks wrong, never stores it."""
    found = details_from_text("my email is rahul@gmail and my number is 98765 4321", AREAS)
    data, errors = LeadData.from_raw(found)
    assert (data.email, data.phone) == (None, None)
    assert set(errors) == {"email", "phone"}


def test_without_the_city_list_only_clear_phrases_give_a_city():
    assert details_from_text("I live in Lucknow", None) == {"city": "Lucknow"}
    assert details_from_text("aap log noida me kaam karte ho", None) == {}


@pytest.mark.parametrize("message, expected", [
    ("I want to go ahead", True),
    ("how do I book a site visit", True),
    ("lagwana hai, aage kya karna hai", True),  # Adithya's example
    ("ready to proceed", True),
    ("what are the next steps?", True),
    ("what is the warranty", False),
    ("do you install in Kanpur?", False),
])
def test_a_clear_buying_signal(message, expected):
    assert is_go_ahead(message) is expected


@pytest.mark.parametrize("message, expected", [
    ("Can someone from Vantis call me?", True),
    ("please call back tomorrow", True),
    ("I'd like a callback", True),
    ("mujhe call karo", True),
    ("can your team contact me?", True),
    ("Do you install in Kanpur?", False),
    ("what does the installation call for?", False),
])
def test_a_request_to_be_called(message, expected):
    assert asks_to_be_called(message) is expected


@pytest.mark.parametrize("message, expected", [
    ("What solar systems do you provide?", True),  # the assignment's Q1
    ("What do you offer?", True),
    ("What packages do you have for homes?", True),
    ("kya kya milta hai", True),
    ("kaunse system hain aapke paas", True),
    ("Tell me the products you sell", True),
    ("what are my financing options", False),  # financing, not the offer
    ("Do you provide installation in Delhi?", False),  # a city question (assignment Q3)
    ("What do you provide in Delhi?", False),
    ("What warranty do you provide?", False),
    ("which service areas do you cover", False),
    ("What is your installation process?", False),
])
def test_a_question_about_what_vantis_offers(message, expected):
    assert asks_what_is_offered(message) is expected


@pytest.mark.parametrize("message, expected", [
    ("Just exploring solar for now, maybe next year",  # D7 turn 1 of the live check
     {"intent_level": "exploring", "timeline": "next year", "timeline_bucket": "gt_6m"}),
    ("abhi bas dekh raha hu, agle saal lagwana hai",
     {"intent_level": "exploring", "timeline": "agle saal", "timeline_bucket": "gt_6m"}),
    ("not now, thanks", {"intent_level": "exploring"}),
    ("next month chahiye", {"timeline": "next month", "timeline_bucket": "lt_1m"}),  # within about a month
    ("agle mahine lagwana hai", {"timeline": "agle mahine", "timeline_bucket": "lt_1m"}),
    ("this month if possible", {"timeline": "this month", "timeline_bucket": "lt_1m"}),
    ("ASAP please", {"timeline": "ASAP", "timeline_bucket": "lt_1m"}),
    ("turant chahiye", {"timeline": "turant", "timeline_bucket": "lt_1m"}),
    ("jaldi lagwana hai", {"timeline": "jaldi", "timeline_bucket": "lt_1m"}),
    ("jaldi batao", {}),  # "tell me quickly" is not a timeline
])
def test_clear_intent_and_timeline_phrases(message, expected):
    assert details_from_text(message, AREAS) == expected


def test_a_name_after_im_right_before_the_phone():
    """D7 turn 2 of the live check."""
    assert details_from_text("I'm Vikram Rao, 9876500004, house in Kanpur, bill around 3000", AREAS) == {
        "name": "Vikram Rao", "phone": "9876500004", "city": "Kanpur", "monthly_bill_inr": "3000"}

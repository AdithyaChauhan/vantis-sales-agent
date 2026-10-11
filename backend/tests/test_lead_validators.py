"""Lead validators: phones, emails, money, kW, kWh, names, and per-field errors in LeadData.from_raw."""
import pytest

from app.leads import validators as v
from app.leads.schemas import LeadData


@pytest.mark.parametrize(
    "raw",
    ["9876543210", "+91 98765 43210", "+91-98765-43210", "098765-43210", "919876543210", "(98765) 43210"],
)
def test_good_phones_become_plus_91(raw):
    assert v.clean_phone(raw) == "+919876543210"


@pytest.mark.parametrize(
    "raw",
    [
        "987654321",  # 9 digits
        "5876543210",  # starts with 5
        "98765abcde",  # letters
        "0522 2345678",  # Lucknow landline
        "+1 415 555 0100",  # not Indian
        "98765432101",  # 11 digits without a 0 prefix
    ],
)
def test_bad_phones_are_rejected(raw):
    with pytest.raises(v.InvalidValue):
        v.clean_phone(raw)


def test_phone_message_can_be_asked_again():
    with pytest.raises(v.InvalidValue, match="10-digit Indian mobile number"):
        v.clean_phone("12345")


def test_email_syntax_and_normalisation():
    assert v.clean_email(" Rahul.Sharma@Example.COM ") == "rahul.sharma@example.com"
    for bad in ["rahul@", "rahul.example.com", "rahul@localhost", "a b@example.com"]:
        with pytest.raises(v.InvalidValue, match="email address"):
            v.clean_email(bad)


def test_email_check_makes_no_network_call(monkeypatch):
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    assert v.clean_email("someone@no-such-domain-vantis-test.com") == "someone@no-such-domain-vantis-test.com"


@pytest.mark.parametrize(
    "raw, rupees",
    [
        ("₹4,000", 4000),
        ("Rs 4000", 4000),
        ("Rs. 4,000/month", 4000),
        ("INR 4000 per month", 4000),
        ("4k", 4000),
        ("4.5K", 4500),
        ("4000/-", 4000),
        (4000, 4000),
        (4000.4, 4000),
        ("10,00,000", 1_000_000),
    ],
)
def test_bill_parsing(raw, rupees):
    assert v.parse_bill(raw) == rupees


def test_budget_accepts_lakh_and_crore():
    assert v.parse_budget("1.2 lakh") == 120_000
    assert v.parse_budget("3L") == 300_000
    assert v.parse_budget("₹1 crore") == 10_000_000


@pytest.mark.parametrize(
    "raw, message",
    [
        ("-4000", "more than zero"),
        ("0", "more than zero"),
        ("4000-5000", "not a range"),
        ("4k to 5k", "not a range"),
        ("about four thousand", "couldn't read"),
        ("50", "between ₹100 and ₹10,00,000"),
        ("20 lakh", "between ₹100 and ₹10,00,000"),
        (True, "as a number"),
    ],
)
def test_bad_bills(raw, message):
    with pytest.raises(v.InvalidValue, match=message):
        v.parse_bill(raw)


@pytest.mark.parametrize("raw", ["5kW", "5 kW", "5 KW", "5 kilowatt", "5", 5, "5.0 kWp"])
def test_kw_parsing(raw):
    assert v.parse_kw(raw) == 5.0


@pytest.mark.parametrize("raw", ["0.1 kW", "500 kW", "-5 kW", "5 MW"])
def test_bad_kw(raw):
    with pytest.raises(v.InvalidValue):
        v.parse_kw(raw)


def test_kwh_parsing():
    assert v.parse_kwh("500 units") == 500
    assert v.parse_kwh("1,200 kWh per month") == 1200
    with pytest.raises(v.InvalidValue, match="between 10 and 100,000"):
        v.parse_kwh("5 units")


def test_names():
    assert v.clean_name("  Rahul   Sharma ") == "Rahul Sharma"
    for bad in ["R", "12345", "x" * 101]:
        with pytest.raises(v.InvalidValue):
            v.clean_name(bad)


def test_from_raw_keeps_valid_values_and_reports_invalid_ones():
    data, errors = LeadData.from_raw(
        {
            "name": "Rahul Sharma",
            "phone": "12345",
            "email": "rahul.sharma@example.com",
            "monthly_bill_inr": "4k",
            "property_type": "castle",
            "favourite_colour": "blue",
            "timeline": "  ",
        }
    )
    assert (data.name, data.phone, data.email, data.monthly_bill_inr) == (
        "Rahul Sharma", None, "rahul.sharma@example.com", 4000
    )
    assert set(errors) == {"phone", "property_type", "favourite_colour"}
    assert "10-digit Indian mobile number" in errors["phone"]
    assert errors["property_type"].startswith("Please choose one of:")
    assert data.timeline is None  # blank text is treated as not given


def test_customer_type_comes_from_the_property_type():
    assert LeadData(property_type="villa").customer_type == "residential"
    assert LeadData(property_type="clinic").customer_type == "commercial"
    assert LeadData(property_type="other").customer_type is None
    assert LeadData(property_type="shop", customer_type="residential").customer_type == "residential"


def test_lead_data_rejects_invalid_values_as_a_whole():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LeadData(phone="12345")  # T9's POST /leads turns this into a 422

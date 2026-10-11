"""Pure validators for lead details. Each returns a clean value or raises InvalidValue.

The messages are plain sentences the agent can turn into a follow-up question.
"""
import re

from email_validator import EmailNotValidError, validate_email

# Limits (docs/SCORING.md). Outside them a value is treated as a mistake, not stored.
BILL_INR = (100, 10_00_000)  # ₹ per month
BUDGET_INR = (10_000, 10_00_00_000)
SIZE_KW = (0.5, 100.0)
UNITS_KWH = (10, 1_00_000)  # per month
NAME_CHARS = (2, 100)
SHORT_TEXT_CHARS = 100  # city, state, business type
TEXT_CHARS = 500  # requirement, timeline, motivation
LONG_TEXT_CHARS = 1000  # other info, issue, project details

_PHONE_JUNK = re.compile(r"[\s\-.()]")
_MULTIPLIERS = {
    None: 1, "k": 1_000, "thousand": 1_000,
    "l": 1_00_000, "lac": 1_00_000, "lacs": 1_00_000, "lakh": 1_00_000, "lakhs": 1_00_000,
    "cr": 1_00_00_000, "crore": 1_00_00_000, "crores": 1_00_00_000,
}
# Currency words and "per month" phrases around an amount
_MONEY_NOISE = re.compile(
    r"₹|\brs\b\.?|\binr\b|\brupees?\b|/-|\bper\s+month\b|/\s*(?:month|mo)\b|\bpm\b|\ba\s+month\b|\bmonthly\b"
)
_RANGE = re.compile(r"\d\s*[a-z]*\s*(?:-|–|\bto\b)\s*\d")  # "4000-5000", "4k to 5k"
_AMOUNT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(k|thousand|l|lacs?|lakhs?|cr|crores?)?")
_KW = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:kw|kwp|kilowatts?)?")
_KWH = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:units?|kwh)?")
_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


class InvalidValue(ValueError):
    """A value that cannot be stored. str(error) is the message for the customer."""


def format_inr(amount: float) -> str:
    """₹ with Indian digit grouping: 1000000 -> ₹10,00,000."""
    whole = str(int(round(amount)))
    if len(whole) <= 3:
        return f"₹{whole}"
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return "₹" + ",".join([head] + groups + [tail])


def clean_phone(value) -> str:
    """Indian mobile number as +91 and 10 digits starting with 6-9.

    Accepts spaces, dashes, dots, brackets and a +91, 91 or 0 prefix. Landlines are rejected.
    """
    text = _PHONE_JUNK.sub("", str(value).strip())
    if text.startswith("+91"):
        text = text[3:]
    elif text.startswith("+"):
        raise InvalidValue("Please share an Indian mobile number (starting with +91).")
    if not text.isdigit():
        raise InvalidValue("A phone number should contain only digits.")
    if len(text) == 12 and text.startswith("91"):
        text = text[2:]
    elif len(text) == 11 and text.startswith("0"):
        text = text[1:]
    if len(text) != 10 or text[0] not in "6789":
        raise InvalidValue(
            "That doesn't look like a 10-digit Indian mobile number (it should start with 6, 7, 8 or 9)."
        )
    return "+91" + text


def clean_email(value) -> str:
    """Email address, checked for syntax only (no network lookup), lowercased."""
    try:
        return validate_email(str(value).strip(), check_deliverability=False).normalized.lower()
    except EmailNotValidError as exc:
        raise InvalidValue(f"That email address doesn't look right: {exc}") from exc


def _number(value, pattern: re.Pattern, what: str, multipliers: dict | None = None) -> float:
    """A number from a value like '4k' or '5 kW'. Rejects ranges, negatives, zero and other text."""
    if isinstance(value, bool):
        raise InvalidValue(f"Please give the {what} as a number.")
    if isinstance(value, (int, float)):
        amount = float(value)
    else:
        text = _MONEY_NOISE.sub(" ", str(value).lower()).replace(",", "").strip()
        if _RANGE.search(text):
            raise InvalidValue(f"Please give one approximate number for the {what}, not a range.")
        match = pattern.fullmatch(text)
        if not match:
            raise InvalidValue(f"I couldn't read the {what} as a number.")
        unit = match.group(2) if multipliers is not None else None
        amount = float(match.group(1)) * (multipliers or {None: 1})[unit]
    if amount <= 0:
        raise InvalidValue(f"The {what} should be more than zero.")
    return amount


def parse_inr(value, *, what: str, limits: tuple[int, int]) -> int:
    """Rupees from '₹4,000', 'Rs 4000', '4k', '1.2 lakh' or a number; must be within limits."""
    amount = _number(value, _AMOUNT, what, _MULTIPLIERS)
    low, high = limits
    if not low <= amount <= high:
        raise InvalidValue(
            f"The {what} should be between {format_inr(low)} and {format_inr(high)}."
        )
    return round(amount)


def parse_bill(value) -> int:
    """Monthly electricity bill in ₹."""
    return parse_inr(value, what="monthly bill", limits=BILL_INR)


def parse_budget(value) -> int:
    """Budget in ₹."""
    return parse_inr(value, what="budget", limits=BUDGET_INR)


def parse_kw(value) -> float:
    """System size in kW from '5kW', '5 kW', '5 kilowatt' or a number."""
    size = _number(value, _KW, "system size")
    if not SIZE_KW[0] <= size <= SIZE_KW[1]:
        raise InvalidValue(f"The system size should be between {SIZE_KW[0]} and {SIZE_KW[1]:g} kW.")
    return round(size, 2)


def parse_kwh(value) -> int:
    """Monthly electricity use in kWh (units) from '500 units', '500 kWh' or a number."""
    units = _number(value, _KWH, "monthly usage")
    if not UNITS_KWH[0] <= units <= UNITS_KWH[1]:
        raise InvalidValue(
            f"The monthly usage should be between {UNITS_KWH[0]} and {UNITS_KWH[1]:,} units (kWh)."
        )
    return round(units)


def clean_name(value) -> str:
    """A person's name: 2-100 characters with at least one letter, spaces collapsed."""
    name = " ".join(str(value).split())
    if not NAME_CHARS[0] <= len(name) <= NAME_CHARS[1] or not any(ch.isalpha() for ch in name):
        raise InvalidValue("Could you share your name?")
    return name


def clean_text(value, *, what: str, max_chars: int = TEXT_CHARS) -> str:
    """Free text: spaces collapsed, not empty, at most max_chars."""
    text = " ".join(str(value).split())
    if not text:
        raise InvalidValue(f"The {what} is empty.")
    if len(text) > max_chars:
        raise InvalidValue(f"The {what} is too long (at most {max_chars} characters).")
    return text


def parse_bool(value, *, what: str) -> bool:
    """True or False from a bool or yes/no text."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _YES:
        return True
    if text in _NO:
        return False
    raise InvalidValue(f"Please answer yes or no for {what}.")

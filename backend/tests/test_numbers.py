"""The number guard (T12b): no number, duration, amount or percentage that nothing in the conversation gave."""
import pytest

from app.agent.welcome import NUMBERS_CONFIRM_TEXT
from app.rag.numbers import known_numbers, number_mentions, remove_unsupported_numbers, unsupported

KB_TIMING = "Installation timing depends on project requirements, site conditions, equipment availability."
CATALOG = ("- Vantis HomeStart 3; Residential; 3 kW rooftop solution\n"
           "- Vantis HomePro 5; Residential; 5 kW rooftop solution")


def guard(reply: str, *allowed: str) -> tuple[str, list[str]]:
    """(reply, removed numbers): the removed sentences are tested on their own below."""
    text, numbers, _ = remove_unsupported_numbers(reply, list(allowed), NUMBERS_CONFIRM_TEXT)
    return text, numbers


@pytest.mark.parametrize("text, found", [
    ("It costs ₹40,000.", ["₹40,000"]),
    ("Panels last 25 years.", ["25 years"]),
    ("You get 30% back.", ["30%"]),
    ("A 25-year warranty.", ["25-year"]),
    ("Support is 24/7.", ["24", "7"]),
    ("It takes a few weeks.", ["a few weeks"]),
    ("About two years, or several months.", ["two years", "several months"]),
    ("Around twenty-five thousand rupees.", ["twenty-five thousand"]),
    ("Panels need cleaning once a year.", ["a year"]),
])
def test_numbers_are_found(text, found):
    assert [m.text for m in number_mentions(text)] == found


@pytest.mark.parametrize("text", [
    "Could you share one more detail?",
    "Share a few details and the team can check.",
    "Vantis currently covers Delhi [KB Section 10: Reference Service Locations].",
    "See KB Section 6 for the steps.",
    "1. Customer Inquiry\n2. System Handover",
])
def test_tags_list_markers_and_number_words_without_units_are_not_numbers(text):
    assert number_mentions(text) == []


def test_a_month_after_an_amount_is_a_rate_not_a_duration():
    assert [m.text for m in number_mentions("For a bill of 2,000 a month, a site assessment comes first.")] == ["2,000"]


def test_groq_drill_sentence_a_few_weeks_is_removed_and_the_team_confirms_instead():
    # T12a slow-Google drill, "How long does installation take?": Groq wrote "the build phase takes a few weeks".
    # The full reply was not kept (logs never hold message text), so it is rebuilt around the quoted sentence.
    reply = ("Installation timing depends on site conditions and equipment availability "
             "[KB Section 6: Installation & Site Assessment]. The build phase takes a few weeks. "
             "Would you like me to check your city?")
    text, removed = guard(reply, KB_TIMING, "How long does installation take?")
    assert removed == ["a few weeks"]
    assert text == ("Installation timing depends on site conditions and equipment availability "
                    f"[KB Section 6: Installation & Site Assessment]. {NUMBERS_CONFIRM_TEXT} "
                    "Would you like me to check your city?")


def test_groq_drill_sentence_with_the_customers_own_size_is_kept():
    # "a 3 kW system is a common size for a house there": 3 kW came from the customer, so the number guard keeps
    # it; a generic claim without an invented number is left to the prompt (a known limit, PLAN.md 8b)
    reply = "Yes, Vantis currently covers Kanpur. A 3 kW system is a common size for a house there."
    assert guard(reply, "Do you install in Kanpur? I'd like about 3 kW for my house.") == (reply, [])


@pytest.mark.parametrize("reply", [
    "That will cost about ₹40,000.",
    "The panels carry a warranty of 25 years.",
    "You could get a 30% subsidy.",
])
def test_invented_price_warranty_and_subsidy_are_replaced(reply):
    text, removed = guard(f"Good question. {reply}", KB_TIMING, "How much will it cost?")
    assert removed and text == f"Good question. {NUMBERS_CONFIRM_TEXT}"


def test_the_fixed_line_is_not_added_when_the_reply_already_says_the_team_confirms():
    reply = "Warranty is usually 25 years. The Vantis team can confirm the terms for your equipment."
    assert guard(reply) == ("The Vantis team can confirm the terms for your equipment.", ["25 years"])


def test_a_count_is_removed_without_the_fixed_line():
    text, removed = guard("The process has 8 steps:\n1. Inquiry\n2. Handover", "General process: Inquiry → Handover")
    assert (text, removed) == ("1. Inquiry\n2. Handover", ["8"])


@pytest.mark.parametrize("reply, allowed", [
    ("For your ₹4,000 bill, a site assessment comes first.", ["my bill is around 4000"]),
    ("For a bill of ₹4,000 a month, a site assessment comes first.", ["my bill is 4k"]),
    ("HomePro 5 is the 5 kW reference package.", [CATALOG]),
    ("Got it: 98765 43210.", ["call me on 9876543210"]),
    ("That doesn't look like a 10-digit Indian mobile number.",
     ["That doesn't look like a 10-digit Indian mobile number (it should start with 6, 7, 8 or 9)."]),
    ("I've logged a callback request for 4pm; the team will confirm it.", ["4pm"]),
    ("Since you'd like it within a month, the team can plan the visit.", ["I want it next month"]),
    ("Your support request id is 12.", ['{"saved": true, "support_request_id": 12}']),
    ("HomeStart 3 is still a reference option.", ["Vantis offers HomeStart 3 (3 kW) and HomePro 5 (5 kW)."]),
])
def test_numbers_from_the_context_tools_customer_or_earlier_replies_are_kept(reply, allowed):
    assert guard(reply, *allowed) == (reply, [])


def test_customer_amounts_are_known_with_and_without_their_multiplier():
    known = known_numbers(["bill 4k, maybe 1.2 lakh budget"])
    assert {4.0, 4000.0, 1.2, 120000.0} <= known.values
    assert unsupported("₹1,20,000 budget and ₹4,000 bill", known) == []


def test_a_list_item_with_an_invented_number_goes_and_the_rest_of_the_list_stays():
    reply = "Steps:\n1. Inquiry\n2. Installation (2 days)\n3. Handover"
    text, removed = guard(reply, "General process: Inquiry → Installation → Handover")
    assert removed == ["2 days"]
    assert text == f"Steps:\n1. Inquiry\n2. {NUMBERS_CONFIRM_TEXT}\n3. Handover"


def test_the_removed_sentences_are_returned_without_tags_and_the_log_has_only_the_count(caplog):
    # T13a: each removed sentence goes into the reply's steps so it can be checked; never into the server log
    reply = ("For your bill, HomeStart 3 could fit [KB Section 9: Guidance]. That saves about 30% "
             "[KB Section 9: Guidance]. Shall I note your city?")
    with caplog.at_level("WARNING", logger="app.rag.numbers"):
        text, numbers, sentences = remove_unsupported_numbers(reply, ["my bill is 4000"], NUMBERS_CONFIRM_TEXT)
    assert sentences == ["For your bill, HomeStart 3 could fit.", "That saves about 30%."]
    assert numbers == ["3", "30%"]
    assert text == f"{NUMBERS_CONFIRM_TEXT} Shall I note your city?"
    [record] = caplog.records
    assert record.removed == 2
    assert "HomeStart" not in record.getMessage() and "30%" not in record.getMessage()
    assert remove_unsupported_numbers("No numbers here.", [], NUMBERS_CONFIRM_TEXT) == ("No numbers here.", [], [])

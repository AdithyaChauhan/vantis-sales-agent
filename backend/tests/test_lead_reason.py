"""The lead reason: LLM sentence when it is safe, template sentence otherwise. Fake LLM only."""
import pytest

from app.core.prompts import load_prompt
from app.leads.reason import template_reason, write_reason
from tests.test_answer import FakeLLM
from tests.test_lead_scoring import scored
from tests.leads_data import RAHUL

GOOD = "Ready buyer in Lucknow with a clear 5 kW requirement, full contact details and a near-term timeline."


@pytest.fixture
def hot():
    return scored(RAHUL)[1]


def test_the_llm_sentence_is_used(hot):
    llm = FakeLLM(GOOD)
    assert write_reason(hot, "Lucknow", llm) == (GOOD, "llm")
    assert len(llm.calls) == 1


def test_quotes_around_the_reply_are_removed(hot):
    assert write_reason(hot, "Lucknow", FakeLLM(f'"{GOOD}"'))[0] == GOOD


@pytest.mark.parametrize(
    "llm",
    [
        FakeLLM(error=RuntimeError("Gemini and Groq both down")),
        FakeLLM(""),
        FakeLLM("A very long reason " * 15 + "."),
        FakeLLM("Ready buyer with a clear need. Contact soon."),  # two sentences
        FakeLLM("Line one\nline two"),
        FakeLLM("This is a COLD lead with weak signals."),  # contradicts HOT
    ],
)
def test_unusable_replies_fall_back_to_the_template(hot, llm):
    reason, source = write_reason(hot, "Lucknow", llm)
    assert source == "template"
    assert reason == template_reason(hot)


@pytest.mark.parametrize(
    "reply",
    [
        "Ready buyer who may be eligible for a subsidy.",
        "Ready buyer interested in low EMI options.",
        "Ready buyer expecting savings on the bill.",
        "Ready buyer asking about financing.",
        "Ready buyer with a guaranteed install slot.",
        "Ready buyer with a ₹4,000 monthly bill.",
        "Ready buyer with a Rs 4000 monthly bill.",
        "Ready buyer comparing prices.",
    ],
)
def test_forbidden_topics_fall_back_to_the_template(hot, reply):
    assert write_reason(hot, "Lucknow", FakeLLM(reply))[1] == "template"


def test_word_starts_only_so_premium_is_not_emi(hot):
    reply = "Ready buyer in Lucknow with a premium rooftop and a clear 5 kW requirement."
    assert write_reason(hot, "Lucknow", FakeLLM(reply)) == (reply, "llm")


def test_the_same_status_word_is_allowed(hot):
    reply = "HOT because the buyer is ready, local and has a clear 5 kW need."
    assert write_reason(hot, "Lucknow", FakeLLM(reply))[1] == "llm"


def test_the_prompt_holds_no_personal_details_or_money(hot):
    llm = FakeLLM(GOOD)
    write_reason(hot, "Lucknow", llm)
    [[message]] = llm.calls
    prompt = message.content
    for secret in ("Rahul", "Sharma", "9876543210", "+91", "@", "example.com", "₹", "Rs", "4000", "4,000"):
        assert secret not in prompt
    assert "monthly bill given" in prompt
    assert "Lead status: HOT (score 97 out of 100)" in prompt
    assert "City: Lucknow" in prompt


def test_prompt_file_has_the_kb_section_16_rule():
    text = load_prompt("lead_qualification_prompt")
    assert "Never mention prices, costs, savings, subsidies, financing, EMIs, guarantees or dates." in text
    for placeholder in ("{status}", "{score}", "{city}", "{factors}", "{caps}"):
        assert placeholder in text


def test_template_for_a_capped_lead():
    _, warm = scored({**RAHUL, "city": "Pune"})
    reason = template_reason(warm)
    assert reason.startswith("WARM: ready to buy")
    assert "capped at WARM" in reason

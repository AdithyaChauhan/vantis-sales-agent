"""The understand step: prompt contents, output handling and failures (fake model, no network)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.understand import (
    ExtractedDetails,
    KbHints,
    Understanding,
    build_prompt,
    format_history,
    load_kb_hints,
    understand,
)
from app.core.llm import LLMError
from app.leads.schemas import LeadData
from tests.agent_fakes import HINTS, Scripted, understood


def test_prompt_has_the_kb_hints_history_and_message():
    history = [HumanMessage(f"question {i}") if i % 2 == 0 else AIMessage(f"answer {i}") for i in range(8)]
    prompt = build_prompt(HINTS, "And in {Kanpur}?", history)
    assert "KB Section 8: Catalog\nKB Section 10: Locations" in prompt
    assert ("[KB Section 8: Catalog]\nAll rows (Offering; Primary Customer; Reference Profile):\n"
            "- Plan A; Residential; 3 kW") in prompt
    # only the last 6 earlier messages
    assert "Customer: question 0" not in prompt and "Assistant: answer 1" not in prompt
    assert "Customer: question 2" in prompt and "Assistant: answer 7" in prompt
    assert "<message>\nAnd in {Kanpur}?\n</message>" in prompt  # braces in the message left alone


def test_prompt_without_kb_hints():
    prompt = build_prompt(KbHints(), "hi", [])
    assert "(not available)" in prompt and "(no earlier messages)" in prompt


def test_long_history_messages_are_cut():
    text = format_history([AIMessage("word " * 300)])
    assert len(text) <= len("Assistant: ") + 500


def test_details_schema_mirrors_lead_data():
    assert list(ExtractedDetails.model_fields) == list(LeadData.model_fields)
    for name, field in ExtractedDetails.model_fields.items():
        assert field.description == LeadData.model_fields[name].description
    # a bad phone number does not break the whole output; LeadData.from_raw reports it later
    assert ExtractedDetails(phone="12345", monthly_bill_inr="4k", system_size_kw=5).phone == "12345"


def test_output_is_turned_into_plain_values():
    llm = Scripted(understood(["question", "question", "buying_interest"], "  catalog  ", system_size_kw=5))
    out = understand(llm, HINTS, "Tell me about HomePro 5", [])
    assert out["intents"] == ["question", "buying_interest"]
    assert out["search_query"] == "catalog"
    assert out["details"] == {"system_size_kw": 5}
    assert out["provider"] == "gemini" and out["latency_ms"] >= 0


def test_blank_query_becomes_none():
    out = understand(Scripted(understood(["greeting"], "   ")), HINTS, "hi", [])
    assert out["search_query"] is None


@pytest.mark.parametrize(
    "llm",
    [
        Scripted(RuntimeError("quota")),
        Scripted({"raw": AIMessage(""), "parsed": None, "parsing_error": ValueError("bad")}),
        Scripted("not a dict"),
    ],
)
def test_failures_raise_llm_error(llm):
    with pytest.raises(LLMError):
        understand(llm, HINTS, "hi", [])


def test_schema_has_the_intents():
    intents = Understanding.model_json_schema()["$defs"]
    assert "ExtractedDetails" in intents
    schema = str(Understanding.model_json_schema())
    for intent in ("reply_to_assistant", "thanks_or_bye", "greeting", "off_topic"):
        assert intent in schema


def test_kb_hints_without_a_database():
    assert load_kb_hints(None) == KbHints()


def test_kb_hints_db_failure_is_not_cached(unreachable_engine):
    assert load_kb_hints(unreachable_engine) == KbHints()

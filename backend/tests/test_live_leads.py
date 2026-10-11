"""Live check of the lead reason: one real fast_llm call. Run: python -m pytest -m live -k leads -s"""
import pytest

from app.core.config import get_settings
from app.core.llm import fast_llm
from app.leads.reason import problem_with, write_reason
from tests.leads_data import RAHUL
from tests.test_lead_scoring import scored

pytestmark = pytest.mark.live


def test_real_reason_for_the_demo_lead():
    settings = get_settings()
    if not (settings.google_api_key or settings.groq_api_key):
        pytest.skip("GOOGLE_API_KEY or GROQ_API_KEY is needed")
    _, breakdown = scored(RAHUL)
    reason, source = write_reason(breakdown, "Lucknow", fast_llm())
    print(f"\n[{source}] {reason}")
    assert source == "llm", "the LLM reply was replaced by the template"
    assert problem_with(reason, "HOT") is None  # one sentence, short, no other status, no KB Section 16 topic

"""Live chat checks: real embeddings, real Neon KB and real LLMs. Reads only.

Run: python -m pytest -m live -s   (-s prints each reply, provider and latency)
Cost: 6 embedding calls, 5 main_llm calls, 1 failing Gemini call and 1 Groq call.
"""
import re

import pytest

from app.core.config import get_settings
from app.core.db import get_engine
from app.core.llm import chat_llm, main_llm
from app.main import app
from app.rag.answer import cited_sections
from app.rag.retriever import count_kb_chunks
from scripts.eval_retrieval import KB_QUESTIONS, load_questions

pytestmark = pytest.mark.live

PDF_QUESTIONS = {q.id: q for q in load_questions(KB_QUESTIONS) if q.id in {"q01", "q02", "q03", "q04", "q05"}}
# A price figure: a currency word or sign, lakh/crore, or a number with thousands separators.
PRICE_RE = re.compile(r"₹|\bRs\.?|\bINR\b|\blakhs?\b|\bcrores?\b|\d{1,3}(?:,\d{2,3})+", re.IGNORECASE)
NUMBER_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|twenty-five|thirty"
YEARS_RE = re.compile(rf"\b(?:\d+|{NUMBER_WORDS})[\s-]*(?:years?|yrs?)\b", re.IGNORECASE)
GREETING_RE = re.compile(r"\s*(?:hello|hi|hey|greetings)\b", re.IGNORECASE)


@pytest.fixture(autouse=True)
def needs_live_setup():
    settings = get_settings()
    if not (settings.database_url and settings.google_api_key):
        pytest.skip("DATABASE_URL and GOOGLE_API_KEY are needed")
    if not count_kb_chunks(get_engine()):
        pytest.skip("kb_chunks is empty; run python -m app.rag.ingest first")


def ask(client, question: str) -> dict:
    response = client.post("/api/chat", json={"message": question})
    assert response.status_code == 200
    body = response.json()
    answer_step = body["steps"][-1]["data"]
    sources = [f"{s['section']} p{s['page']}" for s in body["sources"]]
    print(f"\n[{answer_step['provider']}, {answer_step['latency_ms']} ms] {question}\n{body['reply']}\nsources: {sources}")
    return body


def qid_params() -> list:
    """q01 is a known retrieval miss until T8. Strict: an XPASS fails the run, so the marker gets removed."""
    q01_reason = "needs T8 search_query rewrite, see EVAL.md"
    return [
        pytest.param(qid, marks=pytest.mark.xfail(strict=True, reason=q01_reason)) if qid == "q01" else qid
        for qid in sorted(PDF_QUESTIONS)
    ]


@pytest.mark.parametrize("qid", qid_params())
def test_pdf_question_is_answered_with_a_citation(client, qid):
    question = PDF_QUESTIONS[qid]
    body = ask(client, question.question)
    assert not body["degraded"]
    assert set(cited_sections(body["reply"])) & question.expected, "no expected KB Section cited"
    if qid == "q02":  # 5 kW price
        assert not PRICE_RE.search(body["reply"]), "the reply contains a price"
    if qid == "q05":  # warranty
        assert not YEARS_RE.search(body["reply"]), "the reply states a warranty duration"
        assert not GREETING_RE.match(body["reply"]), "the reply opens with a greeting"


def test_bad_gemini_model_falls_back_to_groq(client):
    settings = get_settings()
    if not (settings.groq_api_key and settings.fallback_model):
        pytest.skip("GROQ_API_KEY and FALLBACK_MODEL are needed")
    app.dependency_overrides[main_llm] = lambda: chat_llm("gemini-model-does-not-exist")
    body = ask(client, PDF_QUESTIONS["q04"].question)
    assert body["steps"][-1]["data"]["provider"] == "groq"
    assert body["reply"] and "<think>" not in body["reply"]
    assert cited_sections(body["reply"]), "Groq reply has no citation"

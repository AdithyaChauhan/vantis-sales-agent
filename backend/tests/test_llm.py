"""Chat-model layer: wiring of Gemini + Groq, provider, latency and reply text. No network calls."""
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableBinding, RunnableLambda, RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.core import llm
from app.core.config import Settings
from app.core.llm import LLMError, chat_llm, invoke_llm, reply_text

MESSAGES = [HumanMessage("hi")]


def settings(**overrides) -> Settings:
    """Settings with fake keys, not read from backend/.env."""
    values = {
        "google_api_key": "fake-google-key",
        "llm_model_main": "gemini-main",
        "llm_model_fast": "gemini-fast",
        "groq_api_key": "fake-groq-key",
        "fallback_model": "groq-model",
    }
    return Settings(_env_file=None, **{**values, **overrides})


def ai(text, provider="google_genai", **kwargs) -> AIMessage:
    return AIMessage(content=text, response_metadata={"model_provider": provider}, **kwargs)


def returns(message):
    return RunnableLambda(lambda _messages: message)


def raises(exc):
    def _fail(_messages):
        raise exc

    return RunnableLambda(_fail)


def test_gemini_with_groq_fallback_settings():
    model = chat_llm("gemini-main", settings())
    assert isinstance(model, RunnableWithFallbacks)

    gemini = model.runnable
    assert isinstance(gemini, RunnableBinding) and isinstance(gemini.bound, ChatGoogleGenerativeAI)
    assert gemini.bound.model == "gemini-main"
    assert gemini.bound.thinking_level == "minimal"
    http = gemini.kwargs["http_options"]
    assert http.timeout == 15_000
    assert http.retry_options.attempts == 2  # the first call + one retry
    assert 429 not in http.retry_options.http_status_codes  # quota errors go straight to Groq
    assert 503 in http.retry_options.http_status_codes

    [groq] = model.fallbacks
    assert isinstance(groq, ChatGroq)
    assert groq.model_name == "groq-model"
    assert groq.request_timeout == 15
    assert groq.max_retries == 0
    assert groq.reasoning_effort == "low"


def test_only_groq_configured():
    model = chat_llm("gemini-main", settings(google_api_key=""))
    assert isinstance(model, ChatGroq)


def test_only_gemini_configured():
    model = chat_llm("gemini-main", settings(groq_api_key=""))
    assert isinstance(model, RunnableBinding) and isinstance(model.bound, ChatGoogleGenerativeAI)


def test_nothing_configured_fails_on_call_not_on_build():
    model = chat_llm("", settings(google_api_key="", groq_api_key=""))
    with pytest.raises(LLMError):
        invoke_llm(model, MESSAGES)


def test_main_and_fast_use_their_own_model_names(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: settings())
    llm.main_llm.cache_clear()
    llm.fast_llm.cache_clear()
    try:
        assert llm.main_llm().runnable.bound.model == "gemini-main"
        assert llm.fast_llm().runnable.bound.model == "gemini-fast"
    finally:
        llm.main_llm.cache_clear()
        llm.fast_llm.cache_clear()


def test_invoke_returns_text_provider_and_latency(caplog):
    with caplog.at_level(logging.INFO, logger="app.core.llm"):
        reply = invoke_llm(returns(ai("  Hello [KB Section 8: Catalog]  ")), MESSAGES)
    assert reply.text == "Hello [KB Section 8: Catalog]"
    assert reply.provider == "gemini"
    assert reply.latency_ms >= 0
    record = next(r for r in caplog.records if r.getMessage() == "llm call")
    assert (record.provider, record.fallback_used) == ("gemini", False)
    assert record.latency_ms == reply.latency_ms


def test_fallback_answer_is_reported_as_groq():
    model = raises(RuntimeError("429 quota")).with_fallbacks([returns(ai("From Groq", "groq"))])
    reply = invoke_llm(model, MESSAGES)
    assert (reply.text, reply.provider) == ("From Groq", "groq")


def test_unknown_provider():
    assert invoke_llm(returns(ai("ok", provider="other")), MESSAGES).provider == "unknown"


def test_failure_becomes_llm_error():
    with pytest.raises(LLMError):
        invoke_llm(raises(TimeoutError("slow")), MESSAGES)


def test_empty_reply_is_an_error():
    with pytest.raises(LLMError):
        invoke_llm(returns(ai("   ")), MESSAGES)


def test_reasoning_never_reaches_the_reply():
    groq_style = ai("The answer.", "groq", additional_kwargs={"reasoning_content": "secret plan"})
    assert reply_text(groq_style) == "The answer."
    inline = ai("<think>secret\nplan</think>\nThe answer.", "groq")
    assert reply_text(inline) == "The answer."
    blocks = ai(
        [{"type": "thinking", "thinking": "secret plan"}, {"type": "text", "text": "The answer."}]
    )
    assert reply_text(blocks) == "The answer."

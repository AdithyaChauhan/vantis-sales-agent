"""Chat-model layer: wiring of Gemini + Groq, provider, latency and reply text. No network calls."""
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.core import llm
from app.core.config import Settings
from app.core.llm import GeminiChat, LLMError, chat_llm, gemini_chat, invoke_llm, provider_of, reply_text

MESSAGES = [HumanMessage("hi")]


class Answer(BaseModel):
    text: str


@tool
def echo(text: str) -> str:
    """Return the text."""
    return text


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


def request_http(model, runnable=None):
    """The http_options GeminiChat sends for a call made through `runnable` (a binding of the model)."""
    kwargs = dict(getattr(runnable, "kwargs", {})) if runnable is not None else {}
    kwargs.pop("ls_structured_output_format", None)  # tracing info; LangChain removes it before the call
    return model._prepare_request(MESSAGES, **kwargs)["config"].http_options


def assert_our_http_rules(http):
    assert http.timeout == 15_000
    assert http.retry_options.attempts == 2  # the first call + one retry
    assert 429 not in http.retry_options.http_status_codes  # quota errors go straight to Groq
    assert 503 in http.retry_options.http_status_codes


def test_gemini_with_groq_fallback_settings():
    model = chat_llm("gemini-main", settings())
    assert isinstance(model, RunnableWithFallbacks)

    gemini = model.runnable
    assert isinstance(gemini, GeminiChat)
    assert gemini.model == "gemini-main"
    assert gemini.thinking_level == "minimal"
    assert_our_http_rules(request_http(gemini))

    [groq] = model.fallbacks
    assert isinstance(groq, ChatGroq)
    assert groq.model_name == "groq-model"
    assert groq.request_timeout == 15
    assert groq.max_retries == 0
    assert groq.reasoning_effort == "low"


def test_http_rules_reach_tool_calls_and_structured_output():
    gemini = gemini_chat("gemini-main", settings())
    tools_binding = gemini.bind_tools([echo])
    assert "tools" in tools_binding.kwargs
    assert_our_http_rules(request_http(gemini, tools_binding))
    structured = gemini.with_structured_output(Answer)
    assert "response_json_schema" in structured.first.kwargs
    assert_our_http_rules(request_http(gemini, structured.first))


def test_tools_on_both_models_and_no_parallel_calls_on_groq():
    model = chat_llm("gemini-main", settings(), tools=[echo])
    assert [t["function"]["name"] for t in model.fallbacks[0].kwargs["tools"]] == ["echo"]
    assert model.fallbacks[0].kwargs["parallel_tool_calls"] is False
    assert "parallel_tool_calls" not in model.runnable.kwargs
    assert "tools" in model.runnable.kwargs


def test_structured_output_on_both_models_keeps_the_raw_message():
    model = chat_llm("gemini-main", settings(), schema=Answer)
    assert isinstance(model, RunnableWithFallbacks)
    for runnable in (model.runnable, model.fallbacks[0]):
        assert "raw" in runnable.output_schema.model_json_schema()["properties"]


def test_only_groq_configured():
    model = chat_llm("gemini-main", settings(google_api_key=""))
    assert isinstance(model, ChatGroq)


def test_only_gemini_configured():
    model = chat_llm("gemini-main", settings(groq_api_key=""))
    assert isinstance(model, GeminiChat)


def test_nothing_configured_fails_on_call_not_on_build():
    model = chat_llm("", settings(google_api_key="", groq_api_key=""))
    with pytest.raises(LLMError):
        invoke_llm(model, MESSAGES)


def test_main_and_fast_use_their_own_model_names(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: settings())
    llm.main_llm.cache_clear()
    llm.fast_llm.cache_clear()
    try:
        assert llm.main_llm().runnable.model == "gemini-main"
        assert llm.fast_llm().runnable.model == "gemini-fast"
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
    assert provider_of(ai("ok", provider="groq")) == "groq"


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

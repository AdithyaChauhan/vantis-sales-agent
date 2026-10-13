"""Chat-model layer: wiring of Gemini + Groq, deadlines, the skip list, provider, latency and reply text.
No network calls."""
import logging
import time

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.core import llm
from app.core.config import Settings
from app.core.llm import (
    Candidate,
    GeminiChat,
    LLMError,
    SkipList,
    answer_candidates,
    chat_llm,
    failure_kind,
    gemini_chat,
    invoke_llm,
    model_chain,
    provider_of,
    reason_candidates,
    reply_text,
    understand_candidates,
)

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
    # Gemini refuses deadlines under 10 s; the shorter cut-off is the wall-clock deadline's job. No retry (T12a).
    assert http.timeout == 10_000
    assert http.retry_options.attempts == 1


def models_in(runnable) -> list:
    """The chat models inside a (tools-bound or structured-output) runnable."""
    return [n.data for n in runnable.get_graph().nodes.values() if isinstance(n.data, (GeminiChat, ChatGroq))]


# ---------- wiring ----------

def test_answer_step_is_gemini_8_s_then_groq_10_s_then_the_fast_model_8_s():
    gemini, groq_model, fast = answer_candidates("gemini-main", settings())
    assert (gemini.key, gemini.use, gemini.seconds) == ("gemini:gemini-main", "answer", 8)
    assert (groq_model.key, groq_model.use, groq_model.seconds) == ("groq:groq-model", "answer", 10)
    assert (fast.key, fast.use, fast.seconds) == ("gemini:gemini-fast", "answer", 8)  # last resort (T12a live run)
    assert isinstance(gemini.model, GeminiChat)
    assert gemini.model.model == "gemini-main" and gemini.model.thinking_level == "minimal"
    assert_our_http_rules(request_http(gemini.model))
    assert isinstance(groq_model.model, ChatGroq)
    assert groq_model.model.model_name == "groq-model"
    assert (groq_model.model.request_timeout, groq_model.model.max_retries) == (10, 0)
    assert groq_model.model.reasoning_effort == "low"


def test_http_rules_reach_tool_calls_and_structured_output():
    gemini = gemini_chat("gemini-main", settings())
    tools_binding = gemini.bind_tools([echo])
    assert "tools" in tools_binding.kwargs
    assert_our_http_rules(request_http(gemini, tools_binding))
    structured = gemini.with_structured_output(Answer)
    assert "response_json_schema" in structured.first.kwargs
    assert_our_http_rules(request_http(gemini, structured.first))


def test_tools_on_every_model_and_no_parallel_calls_on_groq():
    gemini, groq_model, fast = answer_candidates("gemini-main", settings(), tools=[echo])
    assert [t["function"]["name"] for t in groq_model.model.kwargs["tools"]] == ["echo"]
    assert groq_model.model.kwargs["parallel_tool_calls"] is False
    for google in (gemini, fast):
        assert "parallel_tool_calls" not in google.model.kwargs
        assert "tools" in google.model.kwargs


def test_structured_output_on_both_models_keeps_the_raw_message():
    for candidate in answer_candidates("gemini-main", settings(), schema=Answer):
        assert "raw" in candidate.model.output_schema.model_json_schema()["properties"]


def test_only_one_provider_configured():
    [only_groq] = answer_candidates("gemini-main", settings(google_api_key=""))
    assert isinstance(only_groq.model, ChatGroq)
    main, fast = answer_candidates("gemini-main", settings(groq_api_key=""))
    assert (main.key, fast.key) == ("gemini:gemini-main", "gemini:gemini-fast")
    [single] = answer_candidates("gemini-fast", settings(groq_api_key=""))  # the fast model is never tried twice
    assert single.key == "gemini:gemini-fast"


def test_nothing_configured_fails_on_call_not_on_build():
    model = chat_llm("", settings(google_api_key="", groq_api_key=""))
    with pytest.raises(LLMError):
        invoke_llm(model, MESSAGES)


def test_the_reason_uses_the_main_model_for_4_s_and_never_groq():
    [reason] = reason_candidates(settings())
    assert (reason.key, reason.use, reason.seconds) == ("gemini:gemini-main", "reason", 4)
    assert isinstance(reason.model, GeminiChat) and reason.model.model == "gemini-main"
    assert reason_candidates(settings(google_api_key="")) == []  # no Groq: the template answers instead


def test_understand_candidates_are_main_then_groq_then_fast_4_s_each():
    candidates = understand_candidates(Answer, settings())
    [main], [groq_model], [fast] = (models_in(c.model) for c in candidates)
    assert (main.model, groq_model.model_name, fast.model) == ("gemini-main", "groq-model", "gemini-fast")
    assert [c.key for c in candidates] == ["gemini:gemini-main", "groq:groq-model", "gemini:gemini-fast"]
    assert {(c.use, c.seconds) for c in candidates} == {("understand", 4)}
    for gemini in (main, fast):
        assert_our_http_rules(request_http(gemini))
    assert (groq_model.request_timeout, groq_model.max_retries) == (4, 0)
    for candidate in candidates:  # structured output with the raw message kept
        assert "raw" in candidate.model.output_schema.model_json_schema()["properties"]


def test_understand_candidates_leave_out_unconfigured_providers():
    assert len(understand_candidates(Answer, settings(groq_api_key=""))) == 2
    assert len(understand_candidates(Answer, settings(google_api_key=""))) == 1
    assert understand_candidates(Answer, settings(google_api_key="", groq_api_key="")) == []


def test_reason_llm_is_built_from_the_main_model(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: settings())
    built = []
    monkeypatch.setattr(llm, "model_chain", lambda candidates, skips=None: built.append(candidates) or "chain")
    llm.reason_llm.cache_clear()
    try:
        assert llm.reason_llm() == "chain"
        [[reason]] = built
        assert (reason.key, reason.use) == ("gemini:gemini-main", "reason")
    finally:
        llm.reason_llm.cache_clear()


# ---------- invoke_llm and replies ----------

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
    chain = model_chain([Candidate("gemini:m", raises(RuntimeError("down")), "answer", 1),
                         Candidate("groq:m", returns(ai("From Groq", "groq")), "answer", 1)])
    reply = invoke_llm(chain, MESSAGES)
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


# ---------- model_chain: one attempt each, its own deadline, then the next model ----------

def structured(text, provider="google_genai") -> dict:
    """What a structured-output model returns (raw message kept)."""
    return {"raw": ai("", provider), "parsed": Answer(text=text), "parsing_error": None}


def sleeps(seconds, output):
    def _slow(_messages):
        time.sleep(seconds)
        return output

    return RunnableLambda(_slow)


def timed(chain) -> tuple[dict, float]:
    start = time.perf_counter()
    out = chain.invoke(MESSAGES)
    return out, time.perf_counter() - start


def candidate(model, key="gemini:main", use="understand", seconds=0.2) -> Candidate:
    return Candidate(key, model, use, seconds)


def test_slow_model_falls_back_within_the_deadline():
    chain = model_chain([candidate(sleeps(2, structured("late"))),
                         candidate(returns(structured("fallback", "groq")), "groq:m")], SkipList())
    out, elapsed = timed(chain)
    assert out["parsed"].text == "fallback" and provider_of(out["raw"]) == "groq"
    assert elapsed < 1  # not the slow model's 2 s


def test_every_fallback_is_capped_by_its_own_deadline():
    chain = model_chain([candidate(sleeps(2, structured("first")), "a:1"), candidate(sleeps(2, structured("second")), "a:2"),
                         candidate(returns(structured("third")), "a:3")], SkipList())
    out, elapsed = timed(chain)
    assert out["parsed"].text == "third"
    assert elapsed < 2 * 0.2 + 0.3


def test_unparsable_output_falls_through_without_a_skip():
    skips = SkipList()
    broken = {"raw": ai("?"), "parsed": None, "parsing_error": ValueError("bad json")}
    out, _ = timed(model_chain([candidate(returns(broken), "a:1", seconds=1),
                                candidate(returns(structured("parsed")), "a:2", seconds=1)], skips))
    assert out["parsed"].text == "parsed"
    assert skips.left("a:1", "understand") == 0  # bad output is not a capacity problem


def test_all_models_failing_raises_llm_error_from_the_first_error():
    chain = model_chain([candidate(raises(RuntimeError("quota")), "a:1"), candidate(sleeps(2, structured("late")), "a:2")],
                        SkipList())
    with pytest.raises(LLMError) as failed:  # understand() then uses the raw message; the agent the friendly reply
        chain.invoke(MESSAGES)
    assert isinstance(failed.value.__cause__, RuntimeError)
    with pytest.raises(LLMError):  # nothing configured: fails on call, not on build
        model_chain([]).invoke(MESSAGES)


# ---------- the skip list (T12a) ----------

class Clock:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Script:
    """A model whose calls follow a script: "slow" sleeps past the deadline, "ok" answers. Counts calls."""

    def __init__(self, *steps, provider="google_genai"):
        self.steps, self.provider, self.calls = list(steps), provider, 0

    def runnable(self):
        def call(_messages):
            step = self.steps[min(self.calls, len(self.steps) - 1)]
            self.calls += 1
            if step == "slow":
                time.sleep(0.5)
            return structured(step, self.provider)

        return RunnableLambda(call)


GROQ_OK = candidate(returns(structured("from groq", "groq")), "groq:m", seconds=1)


def test_an_understand_timeout_skips_the_model_for_60_s_then_it_is_tried_again():
    clock = Clock()
    skips = SkipList(clock)
    main = Script("slow", "ok")
    chain = model_chain([candidate(main.runnable(), "gemini:main", "understand", 0.1), GROQ_OK], skips)
    assert chain.invoke(MESSAGES)["parsed"].text == "from groq"  # main timed out: the window opens at once
    assert skips.left("gemini:main", "understand") == 60
    _, elapsed = timed(chain)  # inside the window: main is passed over, no wait
    assert main.calls == 1 and elapsed < 0.1
    clock.now += 61
    assert chain.invoke(MESSAGES)["parsed"].text == "ok"  # the window ended: main is tried again
    assert main.calls == 2


def test_one_slow_answer_does_not_open_the_window_but_two_in_a_row_do():
    skips = SkipList(Clock())
    main_answers = Script("slow", "slow")
    main_understands = Script("ok")
    answer = model_chain([candidate(main_answers.runnable(), "gemini:main", "answer", 0.1), GROQ_OK], skips)
    understand = model_chain([candidate(main_understands.runnable(), "gemini:main", "understand", 1)], skips)
    assert answer.invoke(MESSAGES)["parsed"].text == "from groq"  # one slow answer: Groq answers this call
    assert skips.left("gemini:main", "answer") == 0
    understand.invoke(MESSAGES)  # an understand success does not reset the answers' count
    answer.invoke(MESSAGES)  # the 2nd slow answer in a row opens the window
    assert skips.left("gemini:main", "answer") == 60


def test_an_answer_success_resets_the_count():
    skips = SkipList(Clock())
    main = Script("slow", "ok", "slow")
    answer = model_chain([candidate(main.runnable(), "gemini:main", "answer", 0.1), GROQ_OK], skips)
    for _ in range(3):
        answer.invoke(MESSAGES)
    assert skips.left("gemini:main", "answer") == 0


class StatusError(Exception):
    """An API error with an HTTP status, as google-genai (.code) or Groq (.status_code) raise them."""

    def __init__(self, code: int, attribute: str = "code"):
        super().__init__(f"HTTP {code}")
        setattr(self, attribute, code)


@pytest.mark.parametrize("error, kind, seconds", [
    (StatusError(503), "5xx", 120),
    (StatusError(504, "status_code"), "5xx", 120),
    (StatusError(429), "429", 60),
    (StatusError(429, "status_code"), "429", 60),
])
def test_a_5xx_or_429_opens_the_window_at_once_even_for_an_answer(error, kind, seconds):
    skips = SkipList(Clock())
    chain = model_chain([candidate(raises(error), "gemini:main", "answer"), GROQ_OK], skips)
    assert chain.invoke(MESSAGES)["parsed"].text == "from groq"
    assert failure_kind(error) == kind
    # T13c: a 429 or 5xx is about the model itself, so every use skips it
    assert skips.left("gemini:main", "answer") == skips.left("gemini:main", "understand") == seconds


def test_a_400_never_opens_the_window():
    skips = SkipList(Clock())
    chain = model_chain([candidate(raises(StatusError(400)), "gemini:main"), GROQ_OK], skips)
    for _ in range(3):
        chain.invoke(MESSAGES)
    assert skips.left("gemini:main", "understand") == 0


def test_failure_kinds_follow_the_cause_chain():
    request = httpx.Request("POST", "https://example.invalid")

    class Wrapped(Exception):
        pass

    try:  # langchain-google-genai raises its own error "from" the google-genai one
        try:
            raise StatusError(504)
        except StatusError as cause:
            raise Wrapped("server error") from cause
    except Wrapped as exc:
        wrapped = exc
    assert failure_kind(wrapped) == "5xx"
    assert failure_kind(TimeoutError("slow")) == "timeout"
    assert failure_kind(httpx.ReadTimeout("slow", request=request)) == "timeout"
    assert failure_kind(groq.APITimeoutError(request=request)) == "timeout"
    assert failure_kind(ValueError("bad json")) is None


def test_skips_are_logged_without_message_text(caplog):
    skips = SkipList(Clock())
    secret = [HumanMessage("my phone is 9876543210")]
    chain = model_chain([candidate(sleeps(1, structured("late")), "gemini:main", seconds=0.1), GROQ_OK], skips)
    with caplog.at_level(logging.INFO, logger="app.core.llm"):
        chain.invoke(secret)
        chain.invoke(secret)
    messages = [r.getMessage() for r in caplog.records]
    assert "llm marked down" in messages and "llm skipped" in messages
    skipped = next(r for r in caplog.records if r.getMessage() == "llm skipped")
    assert (skipped.model, skipped.use) == ("gemini:main", "understand")
    for record in caplog.records:
        assert "9876543210" not in record.getMessage() and "9876543210" not in str(record.__dict__)


# ---------- T13c: a timeout skips the model for that use only; the error's code in the log ----------

class GroqError(Exception):
    """A Groq API error: status_code, and the JSON body with the error's type and code (groq 0.37.1)."""

    def __init__(self, status: int, error_type: str, code: str, message: str = "Bad request"):
        super().__init__(f"Error code: {status} - {message}")
        self.status_code = status
        self.body = {"error": {"message": message, "type": error_type, "code": code}}


class Fake:
    """A model whose calls follow a script: "slow" sleeps past the deadline, an exception is raised, anything else is
    returned. Counts its calls."""

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), 0

    def runnable(self):
        def call(_messages):
            step = self.steps[min(self.calls, len(self.steps) - 1)]
            self.calls += 1
            if step == "slow":
                time.sleep(0.5)
                return structured("late")
            if isinstance(step, Exception):
                raise step
            return step

        return RunnableLambda(call)


def chains(skips: SkipList, main_answer):
    """The Render sequence of 13 Oct, 08:00-08:01 UTC: understand gets main (timeout), Groq (400) and fast (timeout);
    the answer gets main (`main_answer`), Groq (429) and fast (answers)."""
    understand = {"main": Fake("slow"), "groq": Fake(GroqError(400, "invalid_request_error", "tool_use_failed")),
                  "fast": Fake("slow")}
    answer = {"main": Fake(main_answer), "groq": Fake(GroqError(429, "tokens", "rate_limit_exceeded")),
              "fast": Fake(ai("From fast"))}
    keys = {"main": "gemini:main", "groq": "groq:m", "fast": "gemini:fast"}
    understand_chain = model_chain([candidate(understand[n].runnable(), keys[n], "understand", 0.1) for n in keys], skips)
    answer_chain = model_chain([candidate(answer[n].runnable(), keys[n], "answer", 0.1) for n in keys], skips)
    return understand_chain, answer_chain, understand, answer


def test_the_render_sequence_still_tries_main_and_fast_for_the_answer():
    skips = SkipList(Clock())
    understand_chain, answer_chain, _, answer = chains(skips, "slow")
    with pytest.raises(LLMError):
        understand_chain.invoke(MESSAGES)
    assert skips.left("gemini:main", "understand") == 60 and skips.left("gemini:fast", "understand") == 60
    assert skips.left("gemini:main", "answer") == 0 and skips.left("gemini:fast", "answer") == 0
    assert answer_chain.invoke(MESSAGES).content == "From fast"  # main timed out once, Groq 429, fast answered
    assert (answer["main"].calls, answer["groq"].calls, answer["fast"].calls) == (1, 1, 1)
    assert skips.left("groq:m", "understand") == 60  # the 429 skips Groq for every use; the 400 did not


def test_the_render_sequence_with_main_answering():
    skips = SkipList(Clock())
    understand_chain, answer_chain, _, answer = chains(skips, ai("From main"))
    with pytest.raises(LLMError):
        understand_chain.invoke(MESSAGES)
    assert answer_chain.invoke(MESSAGES).content == "From main"
    assert (answer["groq"].calls, answer["fast"].calls) == (0, 0)


def test_the_render_sequence_through_the_graph_gives_a_slower_reply_not_a_failed_one():
    from tests.agent_fakes import make_graph, run

    skips = SkipList(Clock())
    understand_chain, answer_chain, _, _ = chains(skips, "slow")
    graph, _ = make_graph(understand_chain, answer_chain)
    state = run(graph, "house")
    assert state["steps"][0]["detail"] == "failed, raw message used"
    assert state["reply"] == "From fast"  # before T13c: "try again" (main and fast were skipped for the answer)
    assert state["degraded"] is True  # the understand step failed and used the raw message


def test_a_timeout_skips_the_model_for_that_use_only_and_a_5xx_for_every_use():
    skips = SkipList(Clock())
    skips.failed("gemini:main", "understand", "timeout")
    assert (skips.left("gemini:main", "understand"), skips.left("gemini:main", "answer")) == (60, 0)
    skips.failed("gemini:fast", "understand", "5xx")
    assert skips.left("gemini:fast", "answer") == skips.left("gemini:fast", "reason") == 120


def test_skip_logs_say_which_uses_the_window_covers(caplog):
    skips = SkipList(Clock())
    with caplog.at_level(logging.WARNING, logger="app.core.llm"):
        skips.failed("gemini:main", "understand", "timeout")
        skips.failed("groq:m", "answer", "429")
    assert [r.skip_scope for r in caplog.records if r.getMessage() == "llm marked down"] == ["understand", "all"]


def test_error_detail_gives_the_providers_code_never_the_message():
    groq_error = GroqError(400, "invalid_request_error", "tool_use_failed", message="my phone is 9876543210")
    assert llm.error_detail(groq_error) == {"error_status": 400, "error_type": "invalid_request_error",
                                            "error_code": "tool_use_failed"}

    class GoogleError(Exception):  # google-genai's APIError: code and status
        code, status = 400, "INVALID_ARGUMENT"

    try:  # langchain-google-genai raises its own error "from" the google-genai one
        try:
            raise GoogleError("details with customer text")
        except GoogleError as cause:
            raise RuntimeError("Error calling model") from cause
    except RuntimeError as wrapped:
        assert llm.error_detail(wrapped) == {"error_status": 400, "error_type": "INVALID_ARGUMENT"}
    odd = GroqError(400, "a sentence with spaces", "x" * 80)
    assert llm.error_detail(odd) == {"error_status": 400}  # only identifiers are kept
    assert llm.error_detail(ValueError("bad json")) == {}


def test_a_failed_call_logs_the_providers_code_and_no_message_text(caplog):
    error = GroqError(400, "invalid_request_error", "tool_use_failed", message="my phone is 9876543210")
    chain = model_chain([candidate(raises(error), "groq:m"), GROQ_OK], SkipList(Clock()))
    with caplog.at_level(logging.INFO, logger="app.core.llm"):
        chain.invoke(MESSAGES)
    failed = next(r for r in caplog.records if r.getMessage() == "llm call failed")
    assert (failed.error_status, failed.error_type, failed.error_code) == (400, "invalid_request_error",
                                                                           "tool_use_failed")
    for record in caplog.records:
        assert "9876543210" not in str(record.__dict__)


def test_automatic_function_calling_is_off_for_every_kind_of_call():
    from google.genai import _extra_utils

    gemini = gemini_chat("gemini-main", settings())
    for runnable in (None, gemini.bind_tools([echo]), gemini.with_structured_output(Answer).first):
        kwargs = dict(getattr(runnable, "kwargs", {})) if runnable is not None else {}
        kwargs.pop("ls_structured_output_format", None)
        config = gemini._prepare_request(MESSAGES, **kwargs)["config"]
        assert config.automatic_function_calling.disable is True
        assert _extra_utils.should_disable_afc(config) is True  # the SDK's own check: one request, no AFC loop


# ---------- the patient path for offline scripts (T12b) ----------

def test_patient_retries_a_503_then_answers():
    calls, waits = [], []

    def flaky(_messages):
        calls.append(1)
        if len(calls) == 1:
            raise StatusError(503)
        return structured("second try")

    chain = llm.patient(RunnableLambda(flaky), waits=(5, 20), sleep=waits.append)
    assert chain.invoke(MESSAGES)["parsed"].text == "second try"
    assert (len(calls), waits) == (2, [5])


def test_patient_gives_up_after_its_waits_and_never_retries_a_400():
    waits = []
    with pytest.raises(StatusError):
        llm.patient(raises(StatusError(503)), waits=(5, 20), sleep=waits.append).invoke(MESSAGES)
    assert waits == [5, 20]
    waits.clear()
    with pytest.raises(StatusError):
        llm.patient(raises(StatusError(400)), waits=(5, 20), sleep=waits.append).invoke(MESSAGES)
    assert waits == []


def test_no_skips_never_opens_a_window():
    skips = llm.NoSkips()
    chain = model_chain([candidate(raises(StatusError(503)), "gemini:main", "understand"), GROQ_OK], skips)
    chain.invoke(MESSAGES)
    assert skips.left("gemini:main", "understand") == 0 and llm.SKIPS.left("gemini:main", "understand") == 0


def test_offline_models_have_30_s_deadlines_and_the_chat_keeps_its_own():
    understand = understand_candidates(Answer, settings(), seconds=llm.OFFLINE_TIMEOUT_S)
    assert [c.seconds for c in understand] == [30, 30, 30]
    assert [c.seconds for c in reason_candidates(settings(), seconds=llm.OFFLINE_TIMEOUT_S)] == [30]
    assert [c.seconds for c in understand_candidates(Answer, settings())] == [4, 4, 4]
    assert [c.seconds for c in reason_candidates(settings())] == [4]

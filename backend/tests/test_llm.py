"""Chat-model layer: wiring of Gemini + Groq, deadlines, the skip list, provider, latency and reply text.
No network calls."""
import logging
import threading
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


@pytest.fixture(autouse=True)
def idle_deadline_pool():
    """Wait until every worker of the deadline pool is free. A late fake call of an earlier test keeps its worker until
    its sleep ends, and a deadline counts from submit, so a busy pool turned a fake 429 into a timeout."""
    barrier = threading.Barrier(llm.DEADLINE_WORKERS + 1)
    for _ in range(llm.DEADLINE_WORKERS):
        llm._deadline_pool.submit(barrier.wait, 5)
    barrier.wait(5)


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


def test_the_reason_uses_the_fast_model_for_8_s_and_never_groq():
    """T13d: off the main model's quota; it runs after the reply, so 8 s."""
    [reason] = reason_candidates(settings())
    assert (reason.key, reason.use, reason.seconds) == ("gemini:gemini-fast", "reason", 8)
    assert isinstance(reason.model, GeminiChat) and reason.model.model == "gemini-fast"
    assert reason_candidates(settings(google_api_key="")) == []  # no Groq: the template answers instead
    assert reason_candidates(settings(llm_model_fast="")) == []  # no fast model: the template, never main


def test_understand_candidates_are_main_then_fast_4_s_each_and_never_groq():
    """T13d: Groq's tokens are kept for answers; understand falls back to the raw message and the code's reading."""
    candidates = understand_candidates(Answer, settings())
    [main], [fast] = (models_in(c.model) for c in candidates)
    assert (main.model, fast.model) == ("gemini-main", "gemini-fast")
    assert [c.key for c in candidates] == ["gemini:gemini-main", "gemini:gemini-fast"]
    assert {(c.use, c.seconds) for c in candidates} == {("understand", 4)}
    for gemini in (main, fast):
        assert_our_http_rules(request_http(gemini))
    for candidate in candidates:  # structured output with the raw message kept
        assert "raw" in candidate.model.output_schema.model_json_schema()["properties"]


def test_understand_candidates_leave_out_unconfigured_providers():
    assert len(understand_candidates(Answer, settings(groq_api_key=""))) == 2
    assert understand_candidates(Answer, settings(google_api_key="")) == []  # Groq never understands
    assert len(understand_candidates(Answer, settings(llm_model_fast="gemini-main"))) == 1  # one model, tried once


def test_reason_llm_is_built_from_the_fast_model(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: settings())
    built = []
    monkeypatch.setattr(llm, "model_chain", lambda candidates, skips=None: built.append(candidates) or "chain")
    llm.reason_llm.cache_clear()
    try:
        assert llm.reason_llm() == "chain"
        [[reason]] = built
        assert (reason.key, reason.use) == ("gemini:gemini-fast", "reason")
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


def test_an_understand_timeout_pauses_the_model_for_15_s_then_it_is_tried_again():
    clock = Clock()
    skips = SkipList(clock)
    main = Script("slow", "ok")
    chain = model_chain([candidate(main.runnable(), "gemini:main", "understand", 0.1), GROQ_OK], skips)
    assert chain.invoke(MESSAGES)["parsed"].text == "from groq"  # main timed out: the window opens at once
    assert skips.left("gemini:main", "understand") == 15
    _, elapsed = timed(chain)  # inside the window: main is passed over, no wait
    assert main.calls == 1 and elapsed < 0.1
    clock.now += 16
    assert chain.invoke(MESSAGES)["parsed"].text == "ok"  # the window ended: main is tried again
    assert main.calls == 2


def test_one_slow_answer_pauses_the_model_for_15_s_for_answers_only():
    """T13d: a single timeout pauses the model (it took two in a row before), only for that kind of call."""
    skips = SkipList(Clock())
    main_answers = Script("slow")
    main_understands = Script("ok")
    answer = model_chain([candidate(main_answers.runnable(), "gemini:main", "answer", 0.1), GROQ_OK], skips)
    understand = model_chain([candidate(main_understands.runnable(), "gemini:main", "understand", 1)], skips)
    assert answer.invoke(MESSAGES)["parsed"].text == "from groq"  # one slow answer: Groq answers this call
    assert skips.left("gemini:main", "answer") == 15
    assert skips.left("gemini:main", "understand") == 0
    assert understand.invoke(MESSAGES)["parsed"].text == "ok"  # understand still uses it


def test_a_repeat_right_after_a_pause_doubles_it_up_to_60_s():
    """T13d (Adithya): a long Google slowdown must not cost a doomed deadline on every turn."""
    clock = Clock()
    skips = SkipList(clock)
    pauses = []
    for _ in range(4):  # each failure comes right after the last pause ended
        skips.failed("gemini:main", "answer", "timeout")
        pauses.append(skips.left("gemini:main", "answer"))
        clock.now += pauses[-1] + 1
    assert pauses == [15, 30, 60, 60]
    assert skips.left("gemini:main", "understand") == 0  # still only that kind of call


def test_a_success_or_a_quiet_spell_starts_the_back_off_again():
    clock = Clock()
    skips = SkipList(clock)
    skips.failed("gemini:main", "answer", "5xx")
    clock.now += 16
    skips.failed("gemini:main", "answer", "5xx")
    assert skips.left("gemini:main", "answer") == 30
    clock.now += 31
    skips.succeeded("gemini:main", "answer")
    skips.failed("gemini:main", "answer", "5xx")
    assert skips.left("gemini:main", "answer") == 15  # a success in between
    clock.now += 15 + llm.BACKOFF_GRACE_S + 1
    skips.failed("gemini:main", "answer", "5xx")
    assert skips.left("gemini:main", "answer") == 15  # a quiet spell longer than the grace


def test_an_answer_success_resets_the_count():
    clock = Clock()
    skips = SkipList(clock)
    main = Script("slow", "ok", "slow")
    answer = model_chain([candidate(main.runnable(), "gemini:main", "answer", 0.1), GROQ_OK], skips)
    answer.invoke(MESSAGES)  # slow: paused 15 s
    clock.now += 16
    answer.invoke(MESSAGES)  # ok: the back-off starts again
    clock.now += 1
    answer.invoke(MESSAGES)  # slow again soon after: 15 s, not 30
    assert skips.left("gemini:main", "answer") == 15


class StatusError(Exception):
    """An API error with an HTTP status, as google-genai (.code) or Groq (.status_code) raise them."""

    def __init__(self, code: int, attribute: str = "code"):
        super().__init__(f"HTTP {code}")
        setattr(self, attribute, code)


@pytest.mark.parametrize("error, kind, seconds, understand_left", [
    (StatusError(503), "5xx", 15, 0),
    (StatusError(504, "status_code"), "5xx", 15, 0),
    (StatusError(429), "429", 60, 60),
    (StatusError(429, "status_code"), "429", 60, 60),
])
def test_a_5xx_pauses_that_kind_of_call_for_15_s_and_a_429_every_use(error, kind, seconds, understand_left):
    """T13d: a 5xx says one call was unlucky (15 s, that use only); a 429 is the model's quota (every use)."""
    skips = SkipList(Clock())
    chain = model_chain([candidate(raises(error), "gemini:main", "answer"), GROQ_OK], skips)
    assert chain.invoke(MESSAGES)["parsed"].text == "from groq"
    assert failure_kind(error) == kind
    assert skips.left("gemini:main", "answer") == seconds
    assert skips.left("gemini:main", "understand") == understand_left


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
    assert skips.left("gemini:main", "understand") == 15 and skips.left("gemini:fast", "understand") == 15  # T13d
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


def test_a_timeout_or_5xx_pauses_that_use_only_and_a_429_every_use():
    skips = SkipList(Clock())
    skips.failed("gemini:main", "understand", "timeout")
    assert (skips.left("gemini:main", "understand"), skips.left("gemini:main", "answer")) == (15, 0)
    skips.failed("gemini:fast", "understand", "5xx")
    assert (skips.left("gemini:fast", "understand"), skips.left("gemini:fast", "answer")) == (15, 0)
    skips.failed("groq:m", "answer", "429")
    assert skips.left("groq:m", "answer") == skips.left("groq:m", "reason") == 60
    assert (skips.paused_by("groq:m", "reason"), skips.paused_by("gemini:fast", "understand")) == ("429", "5xx")


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


def test_the_service_tier_is_sent_only_when_it_is_not_standard():
    """T13d (item 7): GEMINI_SERVICE_TIER reaches google-genai's request as serviceTier; standard sends nothing."""
    from google.genai import models as genai_models

    def request_body(tier: str) -> dict:
        gemini = gemini_chat("gemini-main", settings(gemini_service_tier=tier))
        config = gemini._prepare_request(MESSAGES)["config"]
        body = {}
        genai_models._GenerateContentConfig_to_mldev(None, config, body)
        return body

    assert gemini_chat("gemini-main", settings()).service_tier is None
    assert "serviceTier" not in request_body("standard")
    assert request_body("priority")["serviceTier"] == "priority"
    assert request_body(" Flex ")["serviceTier"] == "flex"
    assert "serviceTier" not in request_body("fastest")  # unknown: standard, with a warning


def test_the_last_try_model_is_main_with_a_20_s_deadline_and_a_25_s_http_timeout():
    chain = chat_llm("gemini-main", settings(), tools=[echo])
    assert chain.last_try is not None
    assert llm.LAST_TRY_S == 20 and llm.LAST_TRY_HTTP_S == 25
    patient = gemini_chat("gemini-main", settings(), http_timeout_s=llm.LAST_TRY_HTTP_S)
    assert patient.request_http.timeout == 25_000 and patient.request_http.retry_options.attempts == 1


# ---------- T13d: retry-after, the turn's budget and the last try ----------

class RateLimited(Exception):
    """A provider error with an HTTP response (Groq's 429 carries a retry-after header)."""

    def __init__(self, headers: dict, status: int = 429):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        request = httpx.Request("POST", "https://example.invalid")
        self.response = httpx.Response(status, headers=headers, request=request)


class GeminiQuota(Exception):
    """google-genai's APIError: code and the JSON body in `details`, with a RetryInfo."""

    def __init__(self, delay: str):
        super().__init__("429 RESOURCE_EXHAUSTED")
        self.code = 429
        self.details = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}


def test_retry_after_comes_from_groqs_header_or_geminis_retry_info_within_1_to_60_s():
    assert llm.retry_after_s(RateLimited({"retry-after": "7"})) == 7
    assert llm.retry_after_s(GeminiQuota("13s")) == 13
    assert llm.retry_after_s(GeminiQuota("0.2s")) == 1  # never shorter than 1 s
    assert llm.retry_after_s(RateLimited({"retry-after": "3600"})) == 60  # never longer than 60 s
    assert llm.retry_after_s(RateLimited({})) is None and llm.retry_after_s(ValueError("x")) is None
    try:  # found through the cause chain, like failure_kind
        try:
            raise GeminiQuota("9s")
        except GeminiQuota as cause:
            raise RuntimeError("wrapped") from cause
    except RuntimeError as wrapped:
        assert llm.retry_after_s(wrapped) == 9


def test_a_429_pauses_the_model_for_the_providers_retry_after():
    """Render 13 Oct 12:01:54: Groq's 429 paused it for 60 s; its own retry-after was a few seconds."""
    skips = SkipList(Clock())
    chain = model_chain([candidate(raises(RateLimited({"retry-after": "6"})), "groq:m", "answer"),
                         candidate(returns(structured("from fast")), "gemini:fast", "answer", seconds=1)], skips)
    assert chain.invoke(MESSAGES)["parsed"].text == "from fast"
    assert skips.left("groq:m", "answer") == skips.left("groq:m", "understand") == 6


def test_inside_a_turn_a_model_that_timed_out_is_skipped_for_the_rest_of_it():
    skips = SkipList(Clock())
    main = Script("slow", "ok")
    understand = model_chain([candidate(main.runnable(), "gemini:main", "understand", 0.1), GROQ_OK], skips)
    answer = model_chain([candidate(main.runnable(), "gemini:main", "answer", 0.1), GROQ_OK], skips)
    with llm.turn_scope() as turn:
        understand.invoke(MESSAGES)  # main times out on understand
        assert turn.timed_out == {"gemini:main"} and turn.failures == {"gemini:main": "timeout"}
        assert answer.invoke(MESSAGES)["parsed"].text == "from groq"  # not tried again for the answer
    assert main.calls == 1
    assert skips.left("gemini:main", "answer") == 0  # outside this turn it is only paused for understand
    assert answer.invoke(MESSAGES)["parsed"].text == "ok"  # a new turn (or none) tries it again


def test_inside_a_turn_every_deadline_fits_the_time_left_and_nothing_starts_without_2_s(monkeypatch):
    clock = Clock()
    deadlines = {}
    original = llm._attempt

    def spy(c, messages, config=None, seconds=None, **kwargs):
        deadlines[c.key] = seconds
        return original(c, messages, config=config, seconds=seconds, **kwargs)

    monkeypatch.setattr(llm, "_attempt", spy)
    chain = model_chain([candidate(returns(structured("ok")), "gemini:main", "answer", seconds=10)], SkipList(clock))
    with llm.turn_scope(seconds=3 + llm.TURN_RESERVE_S, clock=clock):
        assert chain.invoke(MESSAGES)["parsed"].text == "ok"
        assert deadlines["gemini:main"] == 3  # 3 s left in the turn, not its own 10 s
        clock.now += 1.5  # 1.5 s left: no model starts
        with pytest.raises(LLMError):
            chain.invoke(MESSAGES)


def last_try_setup(clock, groq_left=None, main_fails=None):
    """Main (timed out this turn), Groq (paused for `groq_left` s by a 429) and the long main for the last try."""
    skips = SkipList(clock, sleep=lambda seconds: setattr(clock, "now", clock.now + seconds))
    if groq_left:
        skips.failed("groq:m", "answer", "429", groq_left)
    groq = candidate(returns(ai("from groq", "groq")), "groq:m", "answer", seconds=1)
    main = candidate(returns(ai("short main")), "gemini:main", "answer", seconds=1)
    patient = candidate(raises(main_fails) if main_fails else returns(ai("patient main")), "gemini:main", "answer",
                        seconds=20)
    return skips, llm.last_try_chain([main, groq], patient, skips)


def test_the_last_try_waits_for_a_model_whose_pause_ends_soon():
    """Render 13 Oct 12:02:48: the turn failed in 1.35 s while Groq's pause had 6 s left."""
    clock = Clock()
    skips, last_try = last_try_setup(clock, groq_left=6)
    with llm.turn_scope(clock=clock) as turn:
        turn.timed_out.add("gemini:main")
        start = clock.now
        assert last_try.invoke(MESSAGES).content == "from groq"
        assert clock.now - start == 6  # it waited for the pause to end, then tried Groq once


def test_otherwise_the_last_try_gives_the_main_model_a_long_deadline():
    clock = Clock()
    _, last_try = last_try_setup(clock, groq_left=30)  # Groq's pause ends too late to wait for it
    with llm.turn_scope(clock=clock) as turn:
        turn.timed_out.add("gemini:main")
        turn.failures["gemini:main"] = "timeout"
        assert last_try.invoke(MESSAGES).content == "patient main"


def test_no_last_try_after_a_429_on_main_without_time_or_outside_a_turn():
    clock = Clock()
    _, last_try = last_try_setup(clock, groq_left=30)
    with pytest.raises(LLMError):  # outside a chat turn (the background reason, offline scripts)
        last_try.invoke(MESSAGES)
    with llm.turn_scope(clock=clock) as turn:
        turn.failures["gemini:main"] = "429"  # its quota: a longer deadline won't help
        with pytest.raises(LLMError):
            last_try.invoke(MESSAGES)
    with llm.turn_scope(clock=clock) as turn:
        clock.now = turn.ends - (llm.LAST_TRY_MIN_S - 1)  # 4 s left
        with pytest.raises(LLMError):
            last_try.invoke(MESSAGES)


def test_a_failed_last_try_counts_like_any_failure():
    clock = Clock()
    skips, last_try = last_try_setup(clock, main_fails=StatusError(503))
    with llm.turn_scope(clock=clock):
        with pytest.raises(StatusError):
            last_try.invoke(MESSAGES)
    assert skips.paused_by("gemini:main", "answer") == "5xx"


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
    assert [c.seconds for c in understand] == [30, 30]
    assert [c.seconds for c in reason_candidates(settings(), seconds=llm.OFFLINE_TIMEOUT_S)] == [30]
    assert [c.seconds for c in understand_candidates(Answer, settings())] == [4, 4]
    assert [c.seconds for c in reason_candidates(settings())] == [8]



def test_a_second_groq_model_follows_the_first_with_its_own_skip_key():
    """T13d round 2: FALLBACK_MODEL_2 has its own tokens a minute, so the first one's 429 doesn't end the chain."""
    keys = [c.key for c in answer_candidates("gemini-main", settings(fallback_model_2="groq-small"))]
    assert keys == ["gemini:gemini-main", "groq:groq-model", "groq:groq-small", "gemini:gemini-fast"]
    second = answer_candidates("gemini-main", settings(fallback_model_2="groq-small"))[2]
    assert (second.model.model_name, second.seconds, second.model.reasoning_effort) == ("groq-small", 10, "low")
    assert len(answer_candidates("gemini-main", settings(fallback_model_2="groq-model"))) == 3  # the same model: once
    assert [c.key for c in understand_candidates(Answer, settings(fallback_model_2="groq-small"))] == [
        "gemini:gemini-main", "gemini:gemini-fast"]  # never for understand


def test_when_the_first_groq_model_is_out_of_tokens_the_second_answers():
    skips = SkipList(Clock())
    chain = model_chain([candidate(raises(RateLimited({"retry-after": "20"})), "groq:big", "answer"),
                         candidate(returns(ai("from the small one", "groq")), "groq:small", "answer", seconds=1)],
                        skips)
    assert chain.invoke(MESSAGES).content == "from the small one"
    assert skips.left("groq:big", "answer") == 20 and skips.left("groq:small", "answer") == 0

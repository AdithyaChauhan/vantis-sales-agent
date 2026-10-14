"""LLM access: the embedder and the chat models. Everything else uses these.

Embedder: google-genai directly. gemini-embedding-2 has no task_type, so the task goes in the text:
documents are "title: ... | text: ...", queries are "task: question answering | query: ...".

Chat models: Gemini (langchain-google-genai) and Groq (langchain-groq). Every chat call goes through
model_chain (T12a): it tries its models in order, one attempt each with a wall-clock deadline, and passes
over a model that failed recently (SKIPS), so a slow Google day costs seconds, not half a minute.
- understand_llm(): main, then fast; 4 s each (T13d: Groq is kept for answers)
- chat_llm(): the answer step; Gemini 8 s, then Groq 10 s, then the second Groq model 10 s (T13d, if set), then the
  fast model 8 s (last resort), plus one last try (chain.last_try) the agent uses when all of them failed
- reason_llm(): the lead reason; the fast model only, 8 s (T13d: off the main model's quota; the caller falls back to
  a template)

T13d: inside a chat turn (turn_scope, set by the chat API) every model call fits the turn's 40 s, a model that timed
out is skipped for the rest of the turn, and the last try waits for a paused model or gives the main model more time.

Offline scripts (eval, seed; T12b) use the patient path instead: offline_understand_llm() and offline_reason_llm()
(30 s per call, no skip list, the whole chain retried after a timeout, 429 or 5xx) and Embedder.embed_queries
(batched, with HTTP retries like ingest). Only the chat stays fail-fast.

Tracing (T13c, on the laptop only): every model attempt, skipped model and query embedding is its own LangSmith run
(langsmith's traceable), nested in the graph step that made it. With LANGSMITH_TRACING off (tests, Render) traceable
just calls the function.
"""
import contextvars
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterator

import groq
import httpx
from google import genai
from google.genai import types
from langchain_core.exceptions import ModelTimeoutError
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 20
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# Ingest: 30 s per call, 1 call + 2 retries.
DOCUMENT_HTTP = types.HttpOptions(
    timeout=30_000,
    retry_options=types.HttpRetryOptions(
        attempts=3, initial_delay=2, max_delay=20, http_status_codes=RETRY_STATUS_CODES
    ),
)
# Chat path (T12a): one attempt, cut off after QUERY_TIMEOUT_S by the deadline pool, so a slow query falls
# back to keyword search within 3 s (it was 10 s + one retry, about 22 s). Gemini refuses deadlines under
# 10 s, so the HTTP timeout stays at 10 s and only ends the late call.
QUERY_HTTP = types.HttpOptions(timeout=10_000, retry_options=types.HttpRetryOptions(attempts=1))
QUERY_TIMEOUT_S = 3


class EmbeddingError(RuntimeError):
    """Embedding failed after retries, or the response had the wrong shape."""


class LLMError(RuntimeError):
    """The chat model failed (every model in the chain failed or was skipped), or gave an empty reply."""


class DeadlineExceeded(LLMError):
    """A call did not answer within its wall-clock deadline."""


# ---------- deadlines and the skip list (T12a) ----------

# A late call keeps its worker busy until its own HTTP timeout (Gemini 10 s, Groq 4-10 s). A normal turn
# holds one worker at a time for about 2 s; a slow-Google turn holds one hung worker for 10 s and then
# skips Gemini. 8 workers keep calls out of the queue (the deadline counts from submit).
DEADLINE_WORKERS = 8
QUEUE_WARN_S = 0.5
# How long a model is paused once its window opens (T13d). A timeout or a server error (503 "high demand", 504) pauses
# it for 15 s, only for the kind of call that failed: one slow or overloaded call says little about the next one. In
# the Render logs of 13 Oct (12:01-12:04 UTC) one 503 paused a model for 120 s for every use and turns failed while
# nothing was being tried. A 429 is a quota, so it pauses every use for the provider's retry-after, else 60 s.
SKIP_SECONDS = {"timeout": 15, "5xx": 15, "429": 60}
MIN_SKIP_S, MAX_SKIP_S = 1, 60  # a provider's retry-after is kept within these
# A timeout or 5xx again within 30 s of a pause ending doubles the pause (15, 30, 60 s), so a long Google slowdown
# doesn't cost a doomed deadline on every turn; a success or a quiet spell starts again at 15 s
BACKOFF_GRACE_S = 30
_TIMEOUT_ERRORS = (TimeoutError, httpx.TimeoutException, groq.APITimeoutError, ModelTimeoutError)
_RETRY_DELAY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s\s*$")  # google.rpc.RetryInfo's retryDelay, e.g. "13s"
# A provider's error type or code is logged only when it looks like an identifier ("tool_use_failed",
# "INVALID_ARGUMENT"), so no message text (which can quote the customer) ever reaches the log (T13c)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

_deadline_pool = ThreadPoolExecutor(max_workers=DEADLINE_WORKERS, thread_name_prefix="llm-deadline")


def call_with_deadline(fn: Callable[[], Any], seconds: float) -> Any:
    """fn() on a small thread pool, raising DeadlineExceeded if it has not returned within `seconds`
    (wall clock). A late call is left to finish there; its own HTTP timeout ends it.

    The HTTP timeout cannot do this: Gemini refuses deadlines under 10 s, and httpx times each network
    phase, not the whole call. This guard can also be tested with a fake slow model.
    """
    queued = time.perf_counter()

    def run():
        waited = time.perf_counter() - queued
        if waited > QUEUE_WARN_S:  # the deadline counts from here too: the pool is too small
            logger.warning("llm call waited for a worker", extra={"latency_ms": round(waited * 1000)})
        return fn()

    future = _deadline_pool.submit(contextvars.copy_context().run, run)  # keeps tracing context
    try:
        return future.result(timeout=seconds)
    except FutureTimeout:  # the deadline passed (or the call itself timed out)
        raise DeadlineExceeded(f"no answer within {seconds} s") from None


def failure_kind(exc: BaseException) -> str | None:
    """'timeout', '429' or '5xx' when the error (or one it was raised from) is one of those; None for
    anything else (a 400, a missing model, unparsable output), which never opens a skip window."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (DeadlineExceeded, *_TIMEOUT_ERRORS)):
            return "timeout"
        code = getattr(exc, "status_code", None)  # Groq (and httpx)
        if not isinstance(code, int):
            code = getattr(exc, "code", None)  # google-genai
        if isinstance(code, int) and code == 429:
            return "429"
        if isinstance(code, int) and 500 <= code <= 599:
            return "5xx"
        exc = exc.__cause__ or exc.__context__
    return None


def error_detail(exc: BaseException) -> dict:
    """The provider's HTTP status and its error type or code, from the error or one it was raised from, never the
    message text (T13c: Groq's "400 Bad Request" on understand showed no cause). Groq keeps them in the JSON body
    ({"error": {"type", "code"}}); google-genai has `code` (the status) and `status` (e.g. INVALID_ARGUMENT).
    {} when no error in the chain came from a provider."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        status = getattr(exc, "status_code", None)  # Groq (and httpx)
        if isinstance(status, int):
            body = getattr(exc, "body", None)
            error = body.get("error") if isinstance(body, dict) else None
            found = {"error_status": status}
            if isinstance(error, dict):
                found |= {f"error_{key}": str(error[key]) for key in ("type", "code")
                          if error.get(key) and _IDENTIFIER_RE.match(str(error[key]))}
            return found
        code = getattr(exc, "code", None)  # google-genai
        if isinstance(code, int):
            status_name = getattr(exc, "status", None)
            ok = isinstance(status_name, str) and _IDENTIFIER_RE.match(status_name)
            return {"error_status": code, **({"error_type": status_name} if ok else {})}
        exc = exc.__cause__ or exc.__context__
    return {}


def retry_after_s(exc: BaseException) -> float | None:
    """How long the provider asked us to wait, from the error or one it was raised from (T13d), kept within
    MIN_SKIP_S-MAX_SKIP_S; None when it gave no time. Groq (and any HTTP error) sends a `retry-after` header; Gemini puts
    a google.rpc.RetryInfo with `retryDelay` ("13s") in the error's details."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        found = _header_retry_after(getattr(exc, "response", None)) or _retry_info_delay(getattr(exc, "details", None))
        if found is not None:
            return min(MAX_SKIP_S, max(MIN_SKIP_S, found))
        exc = exc.__cause__ or exc.__context__
    return None


def _header_retry_after(response: Any) -> float | None:
    """The `retry-after` header in seconds, or None (no response, no header, or a date instead of a number)."""
    headers = getattr(response, "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def _retry_info_delay(details: Any) -> float | None:
    """retryDelay from a google-genai error body: {"error": {"details": [{"@type": "...RetryInfo",
    "retryDelay": "13s"}]}}."""
    error = details.get("error", details) if isinstance(details, dict) else None
    items = error.get("details") if isinstance(error, dict) else None
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            match = _RETRY_DELAY_RE.match(str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None


def failures_before_skip(use: str) -> int:
    """How many failures in a row of one model and use open its window: one (T13d: a single 503, 504 or timeout pauses
    the model for 15 s), except the query embedder, which needs 2, because keyword-only search finds worse answers."""
    return 2 if use == "embed" else 1


def skip_scope(use: str, kind: str) -> str | None:
    """Which uses a failure's window covers: a 429 (a quota for the whole model) every use (None); a timeout or a 5xx
    only the use that failed (T13c for timeouts, T13d for 5xx: one 503 on understand had paused the main model for the
    answers too, Render logs of 13 Oct, 12:01:30 UTC)."""
    return None if kind == "429" else use


class SkipList:
    """Models that failed recently and are paused until their window ends (in memory, one process, like the rate
    limiter). A timeout's or 5xx's window covers its use only, a 429's every use (skip_scope). A repeat soon after a
    pause doubles it (BACKOFF_GRACE_S). The count of failures in a row and the back-off are per model and use, and a
    success of that use resets both. `clock` and `sleep` are injectable, so tests run on a fake clock."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self._clock = clock
        self.sleep = sleep  # the last try waits for a paused model with it
        # (model key, use or None for every use) -> (clock time the pause ends, the failure kind that opened it)
        self._until: dict[tuple[str, str | None], tuple[float, str]] = {}
        self._streak: dict[tuple[str, str], int] = {}  # (model key, use) -> failures in a row
        self._backoff: dict[tuple[str, str], tuple[float, float]] = {}  # (key, use) -> (last pause length, its end)
        self._lock = threading.Lock()

    def _open(self, key: str, use: str) -> tuple[float, str] | None:
        """The open window for this use that ends last, as (end, kind), or None."""
        now = self._clock()
        found = [w for w in (self._until.get((key, None)), self._until.get((key, use))) if w and w[0] > now]
        return max(found) if found else None

    def left(self, key: str, use: str) -> float:
        """Seconds until the model may be tried again for this use (0: try it now)."""
        with self._lock:
            window = self._open(key, use)
            return max(0.0, window[0] - self._clock()) if window else 0.0

    def paused_by(self, key: str, use: str) -> str | None:
        """The failure kind ('timeout', '429' or '5xx') whose window is open for this use, or None."""
        with self._lock:
            window = self._open(key, use)
            return window[1] if window else None

    def failed(self, key: str, use: str, kind: str, seconds: float | None = None) -> bool:
        """Count a timeout, 429 or 5xx; open the model's window once enough happened in a row. `seconds` is the
        provider's retry-after, used instead of the usual pause when given. True if a window opened."""
        scope = skip_scope(use, kind)
        with self._lock:
            now = self._clock()
            streak = self._streak.get((key, use), 0) + 1
            if streak < failures_before_skip(use):
                self._streak[(key, use)] = streak
                return False
            self._streak[(key, use)] = 0
            pause = float(seconds) if seconds is not None else float(SKIP_SECONDS[kind])
            if kind in ("timeout", "5xx") and seconds is None:
                last = self._backoff.get((key, use))
                if last and now - last[1] <= BACKOFF_GRACE_S:  # failed again right after (or during) its last pause
                    pause = min(MAX_SKIP_S, max(pause, last[0] * 2))
                self._backoff[(key, use)] = (pause, now + pause)
            self._until[(key, scope)] = (now + pause, kind)
        logger.warning("llm marked down", extra={"model": key, "use": use, "skip_reason": kind,
                                                 "skip_s": round(pause, 1), "skip_scope": scope or "all"})
        return True

    def succeeded(self, key: str, use: str) -> None:
        """A call of this use answered: its count of failures in a row and its back-off start again."""
        with self._lock:
            self._streak.pop((key, use), None)
            self._backoff.pop((key, use), None)

    def clear(self) -> None:
        """Forget every failure (tests)."""
        with self._lock:
            self._until.clear()
            self._streak.clear()
            self._backoff.clear()


SKIPS = SkipList()


class NoSkips(SkipList):
    """The offline scripts' skip list: never skips a model. They wait and retry instead (patient), because a
    script has no customer waiting and a skipped model would only make the next retry fail at once."""

    def left(self, key: str, use: str) -> float:
        return 0.0

    def failed(self, key: str, use: str, kind: str, seconds: float | None = None) -> bool:
        return False


# ---------- the turn's time budget (T13d) ----------

# A chat turn ends within about 40 s: model calls must be done 3 s before that (finalize, lead writes, the checkpoint),
# and no model call starts with less than 2 s left
TURN_SECONDS = 40
TURN_RESERVE_S = 3
MIN_ATTEMPT_S = 2
# The last try, when every answer model failed or was paused and no tool wrote anything: wait up to 10 s for a paused
# model, or give the main model 20 s (Google answered "ok" in 17-25 s on the audit's slow day, so its 8 s deadline was
# hopeless); never with less than 5 s left. Its Gemini HTTP timeout is 25 s, so the deadline cuts it first.
LAST_TRY_WAIT_MAX_S = 10
LAST_TRY_S = 20
LAST_TRY_HTTP_S = 25
LAST_TRY_MIN_S = 5


@dataclass
class TurnBudget:
    """One chat turn's time for model calls (T13d), shared by every model call of the turn: when it ends, and which
    models timed out (skipped for the rest of the turn) or failed in it."""

    ends: float  # clock time by which every model call must be done
    clock: Callable[[], float] = time.monotonic
    timed_out: set[str] = field(default_factory=set)  # model keys
    failures: dict[str, str] = field(default_factory=dict)  # model key -> its last failure kind in this turn

    def left(self) -> float:
        """Seconds left for model calls in this turn."""
        return max(0.0, self.ends - self.clock())


_TURN: contextvars.ContextVar[TurnBudget | None] = contextvars.ContextVar("llm_turn", default=None)


@contextmanager
def turn_scope(seconds: float = TURN_SECONDS, clock: Callable[[], float] = time.monotonic) -> Iterator[TurnBudget]:
    """Run one chat turn inside a time budget (the chat API wraps agent.invoke in it). Every model call of the turn
    sees it, also in the graph's and the deadline pool's threads (they copy the context; the budget is one object).
    Outside a turn (the background reason, offline scripts) model calls work as before."""
    token = _TURN.set(TurnBudget(ends=clock() + seconds - TURN_RESERVE_S, clock=clock))
    try:
        yield _TURN.get()
    finally:
        _TURN.reset(token)


def current_turn() -> TurnBudget | None:
    """The running turn's budget, or None outside a chat turn."""
    return _TURN.get()


# Offline scripts (T12b): 30 s per call, the whole chain tried 3 times, 5 s then 20 s apart
OFFLINE_TIMEOUT_S = 30
OFFLINE_WAITS_S = (5, 20)


def patient(chain: Runnable, waits: tuple[float, ...] = OFFLINE_WAITS_S,
            sleep: Callable[[float], None] = time.sleep) -> Runnable:
    """`chain`, tried again after a timeout, 429 or 5xx (failure_kind) once per wait; any other error (a 400, a
    missing model) is raised at once. For offline scripts only: the chat never waits like this."""

    def call(messages, config: RunnableConfig) -> Any:
        for attempt in range(len(waits) + 1):
            try:
                return chain.invoke(messages, config)
            except Exception as exc:
                kind = failure_kind(exc)
                if attempt == len(waits) or kind is None:
                    raise
                logger.warning("offline llm retry", extra={"attempt": attempt + 1, "failure": kind,
                                                            "wait_s": waits[attempt]})
                sleep(waits[attempt])
        raise AssertionError("unreachable")

    return RunnableLambda(call, name="patient")


# ---------- tracing (T13c) ----------

def note_failure(exc: BaseException) -> None:
    """Add why a call failed (timeout, 429, 5xx or the error's type, plus error_detail) to the current LangSmith run's
    metadata. Does nothing when tracing is off."""
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata({"failure": failure_kind(exc) or type(exc).__name__, **error_detail(exc)})


def _embedding_inputs(inputs: dict) -> dict:
    return {"text": inputs.get("text")}


def _embedding_outputs(vector: Any) -> dict:
    return {"dimensions": len(vector) if isinstance(vector, list) else None}


@traceable(run_type="embedding", name="query embedding", process_inputs=_embedding_inputs,
           process_outputs=_embedding_outputs)
def _traced_embedding(embed: Callable[[], list[float]], text: str) -> list[float]:
    """embed(), as its own LangSmith run (the question, the vector's length, the failure reason)."""
    try:
        return embed()
    except Exception as exc:
        note_failure(exc)
        raise


# ---------- embeddings ----------

def query_text(query: str) -> str:
    """Wrap a user question in the query format gemini-embedding-2 expects."""
    return f"task: question answering | query: {query}"


class Embedder:
    """Turns texts into vectors of exactly `dim` numbers."""

    def __init__(self, client, model: str, dim: int):
        self.client = client
        self.model = model
        self.dim = dim
        self._query_cache = lru_cache(maxsize=256)(self._embed_query_uncached)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts in batches of 20. Returns one vector per text, in order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors += self._embed(batch, DOCUMENT_HTTP)
        return vectors

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed questions for an offline script (the eval, T12b): batches of 20 with ingest's patient HTTP options
        (30 s, 3 attempts on 429 and 5xx), no skip list and no cache. The chat uses embed_query (fail-fast).
        A Google 503 stopped a T12a eval run that embedded through the chat path."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors += self._embed([query_text(t) for t in batch], DOCUMENT_HTTP)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one question (cached, so a repeated question costs no quota)."""
        return list(self._query_cache(text))

    def _embed_query_uncached(self, text: str) -> tuple[float, ...]:
        """One attempt, cut off after QUERY_TIMEOUT_S; skipped after 2 failures in a row (T12a). Traced as its own
        run when tracing is on (T13c)."""
        key = f"gemini-embed:{self.model}"
        left = SKIPS.left(key, "embed")
        if left:
            logger.info("llm skipped", extra={"model": key, "use": "embed", "skip_left_s": round(left)})
            raise EmbeddingError(f"query embedding skipped for {round(left)} s")
        try:
            vector = _traced_embedding(
                lambda: call_with_deadline(lambda: self._embed([query_text(text)], QUERY_HTTP)[0], QUERY_TIMEOUT_S),
                text, langsmith_extra={"name": f"embed: {key}", "metadata": {"model": key, "use": "embed",
                                                                            "deadline_s": QUERY_TIMEOUT_S}})
        except Exception as exc:
            kind = failure_kind(exc)
            if kind:
                SKIPS.failed(key, "embed", kind, retry_after_s(exc))
            if isinstance(exc, EmbeddingError):
                raise
            raise EmbeddingError(f"query embedding failed: {type(exc).__name__}") from exc
        SKIPS.succeeded(key, "embed")
        return tuple(vector)

    def _embed(self, texts: list[str], http_options: types.HttpOptions) -> list[list[float]]:
        """One API call. Each text gets its own Content, otherwise the API merges them into one vector."""
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        config = types.EmbedContentConfig(
            output_dimensionality=self.dim, http_options=http_options
        )
        try:
            response = self.client.models.embed_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:
            raise EmbeddingError(f"embedding call failed: {type(exc).__name__}: {exc}") from exc
        vectors = [list(e.values or []) for e in (response.embeddings or [])]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        for vector in vectors:
            if len(vector) != self.dim:
                raise EmbeddingError(f"expected {self.dim} numbers per vector, got {len(vector)}")
        return vectors


@lru_cache
def embedder() -> Embedder:
    """The shared embedder. Raises EmbeddingError if the key or model name is not set."""
    settings = get_settings()
    if not settings.google_api_key or not settings.embedding_model:
        raise EmbeddingError("GOOGLE_API_KEY and EMBEDDING_MODEL must be set in backend/.env")
    client = genai.Client(api_key=settings.google_api_key)
    return Embedder(client, settings.embedding_model, settings.embedding_dim)


def query_embedder() -> Embedder | None:
    """embedder() for request handlers: None instead of raising when Gemini is not configured.

    search() then tries again, fails, and falls back to keyword-only search. Tests override this.
    """
    try:
        return embedder()
    except EmbeddingError:
        return None


# ---------- chat models ----------

# The Gemini API refuses a deadline under 10 s (400 "Minimum allowed deadline is 10s"), so every Gemini chat
# request gets a 10 s HTTP timeout and no retry; call_with_deadline cuts the call off sooner (T12a).
GEMINI_MIN_TIMEOUT_S = 10
GEMINI_CHAT_HTTP = types.HttpOptions(
    timeout=GEMINI_MIN_TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)
)
# Automatic function calling off (T13c). google-genai runs it by default and logs "AFC is enabled with max remote
# calls: 10" on every call. It can only run Python functions passed as tools, and LangChain passes function
# declarations, so it never ran anything; turned off, the SDK makes exactly one request per call and the
# graph's ToolNode stays the only thing that runs tools.
AFC_OFF = types.AutomaticFunctionCallingConfig(disable=True)
# Wall-clock deadlines per use (T12a). Understand is a small call (the main model answered it in at most 1.9 s in
# T10b, so 4 s is twice that); an answer is longer and may call tools. Groq is the answer's last resort, so it gets
# more room. The reason runs after the reply (or offline), so it can wait longer (T13d: 8 s, was 4 s on main).
UNDERSTAND_TIMEOUT_S = 4
ANSWER_TIMEOUT_S = 8
GROQ_ANSWER_TIMEOUT_S = 10
REASON_TIMEOUT_S = 8
# GEMINI_SERVICE_TIER (T13d): "standard" sends nothing (the API's default); "priority" needs a Tier 2/3 paid project
# and costs more, "flex" is cheaper and slower
GEMINI_SERVICE_TIERS = ("standard", "priority", "flex")
THINKING_LEVEL = "minimal"  # Gemini 3.x thinks by default; minimal is the lowest level (fastest)
GROQ_MAX_RETRIES = 0
GROQ_REASONING_EFFORT = "low"  # keeps the fallback fast and under Groq's 8K tokens/min
PROVIDERS = {"google_genai": "gemini", "groq": "groq"}  # response_metadata["model_provider"] -> name
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class LLMReply:
    """The answer text, which provider wrote it (gemini | groq | unknown) and how long it took."""

    text: str
    provider: str
    latency_ms: int


class GeminiChat(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI that sends our timeout and retry rules (request_http, by default
    GEMINI_CHAT_HTTP) and automatic function calling off (AFC_OFF, T13c) with every request.

    bind_tools() drops bound http_options and with_structured_output() rejects them, so they are added
    here, where every call (plain, tools, structured output) builds its request. This relies on a private
    method of langchain-google-genai 4.4.1 (pinned; tests/test_llm.py checks it).
    """

    request_http: Any = GEMINI_CHAT_HTTP
    service_tier: str | None = None  # T13d: sent only when set ("priority" or "flex"; standard is the default)

    def _prepare_request(self, messages, **kwargs):
        kwargs.setdefault("http_options", self.request_http)
        kwargs.setdefault("automatic_function_calling", AFC_OFF)
        if self.service_tier:  # google-genai's GenerateContentConfig.service_tier, sent as serviceTier
            kwargs.setdefault("service_tier", self.service_tier)
        return super()._prepare_request(messages, **kwargs)


def gemini_service_tier(settings: Settings) -> str | None:
    """GEMINI_SERVICE_TIER for the request: None for "standard" (nothing is sent) or an unknown value (logged)."""
    tier = (settings.gemini_service_tier or "standard").strip().lower()
    if tier not in GEMINI_SERVICE_TIERS:
        logger.warning("GEMINI_SERVICE_TIER is not one of %s; using standard", ", ".join(GEMINI_SERVICE_TIERS))
        return None
    return None if tier == "standard" else tier


def gemini_chat(model: str, settings: Settings, http_timeout_s: float = GEMINI_MIN_TIMEOUT_S) -> GeminiChat:
    """Gemini chat model: minimal thinking, a 10 s HTTP timeout (T13d: longer for the last try) and no retry."""
    http = GEMINI_CHAT_HTTP if http_timeout_s == GEMINI_MIN_TIMEOUT_S else types.HttpOptions(
        timeout=int(http_timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1))
    return GeminiChat(model=model, google_api_key=settings.google_api_key, thinking_level=THINKING_LEVEL,
                      request_http=http, service_tier=gemini_service_tier(settings))


def groq_chat(settings: Settings, timeout_s: float, model: str | None = None) -> ChatGroq:
    """A Groq fallback model (FALLBACK_MODEL, or `model`: FALLBACK_MODEL_2). One attempt; its reasoning never reaches
    the reply. Both must accept reasoning_effort (gpt-oss does)."""
    return ChatGroq(
        model=model or settings.fallback_model,
        api_key=settings.groq_api_key,
        timeout=timeout_s,
        max_retries=GROQ_MAX_RETRIES,
        reasoning_effort=GROQ_REASONING_EFFORT,
    )


def _shape(model: Runnable, provider: str, tools: list | None, schema: type | None) -> Runnable:
    """The model as plain chat, with tools bound, or with structured output (raw message kept)."""
    if tools:
        # gpt-oss on Groq does not support parallel tool calls
        extra = {"parallel_tool_calls": False} if provider == "groq" else {}
        return model.bind_tools(tools, **extra)
    if schema is not None:
        return model.with_structured_output(schema, include_raw=True)
    return model


@dataclass(frozen=True)
class Candidate:
    """One model to try: its key in the skip list ('gemini:<model>' or 'groq:<model>'), the model (plain,
    tools bound or structured output), its use (understand, answer or reason) and its wall-clock deadline."""

    key: str
    model: Runnable
    use: str
    seconds: float


def tokens_of(result: Any) -> int | None:
    """Total tokens of a call (input and output) from the reply's usage metadata, if the provider sent it."""
    message = result.get("raw") if isinstance(result, dict) else result
    usage = getattr(message, "usage_metadata", None) or {}
    return usage.get("total_tokens")


def _attempt_inputs(inputs: dict) -> dict:
    """An attempt run's input in LangSmith: the prompt messages (not the model object or the config)."""
    return {"messages": inputs.get("messages")}


def _attempt_outputs(result: Any) -> dict:
    """An attempt run's output in LangSmith: the answer (structured: the parsed value) and its total tokens."""
    if isinstance(result, dict) and "parsed" in result:
        parsed = result["parsed"]
        output = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
    else:
        output = result
    return {"output": output, "tokens": tokens_of(result)}


@traceable(name="llm attempt", process_inputs=_attempt_inputs, process_outputs=_attempt_outputs)
def _attempt(candidate: Candidate, messages: Any, config: RunnableConfig | None = None,
             seconds: float | None = None) -> Any:
    """One model's attempt, cut off at its deadline (`seconds`, default the candidate's); raises on any failure,
    including unparsable structured output.

    With tracing on (T13c) it is a LangSmith run named "<use>: <model key>", with the deadline in its metadata and,
    when it fails, the reason (note_failure); the provider's own run (prompt, output, tokens) nests inside it."""
    try:
        result = call_with_deadline(lambda: candidate.model.invoke(messages, config), seconds or candidate.seconds)
        if isinstance(result, dict) and "parsed" in result and result["parsed"] is None:
            raise LLMError("structured output could not be parsed")
    except Exception as exc:
        note_failure(exc)
        raise
    return result


@traceable(name="llm skipped")
def _skipped(key: str, use: str, left_s: int) -> dict:
    """A model passed over because of its skip window, as a LangSmith run, so a trace shows why it was not tried."""
    return {"skipped_for_s": left_s}


def _run(c: Candidate, messages: Any, config: RunnableConfig | None, skip_list: SkipList, seconds: float) -> Any:
    """One logged attempt of a candidate with this deadline. A timeout, 429 or 5xx goes to the skip list (with the
    provider's retry-after) and, inside a turn, to the turn's record (T13d). Raises what the attempt raised."""
    run = {"name": f"{c.use}: {c.key}", "metadata": {"model": c.key, "use": c.use, "deadline_s": seconds}}
    start = time.perf_counter()
    try:
        result = _attempt(c, messages, config=config, seconds=seconds, langsmith_extra=run)
    except Exception as exc:
        kind = failure_kind(exc)
        logger.warning("llm call failed", extra={
            "model": c.key, "use": c.use, "failure": kind or type(exc).__name__,
            "latency_ms": round((time.perf_counter() - start) * 1000), **error_detail(exc),
        })
        if kind:
            skip_list.failed(c.key, c.use, kind, retry_after_s(exc))
            turn = current_turn()
            if turn is not None:
                turn.failures[c.key] = kind
                if kind == "timeout":
                    turn.timed_out.add(c.key)
        raise
    skip_list.succeeded(c.key, c.use)
    logger.info("llm call", extra={
        "model": c.key, "use": c.use, "latency_ms": round((time.perf_counter() - start) * 1000),
        "tokens": tokens_of(result),
    })
    return result


def _pass_over(c: Candidate, why: str, left_s: float | None, first_error: Exception | None) -> Exception:
    """Log (and trace) a candidate that is not tried; returns the chain's first error."""
    extra = {"model": c.key, "use": c.use}
    extra |= {"skip_left_s": round(left_s)} if left_s is not None else {"skip_reason": why}
    logger.info("llm skipped", extra=extra)
    run = {"name": f"{c.use}: {c.key} (skipped)", "metadata": {"model": c.key, "use": c.use, "why": why}}
    _skipped(c.key, c.use, round(left_s or 0), langsmith_extra=run)
    return first_error or LLMError(f"{c.key} skipped: {why}")


def model_chain(candidates: list[Candidate], skips: SkipList | None = None) -> Runnable:
    """Try the candidates in order, one attempt each with its own deadline, and return the first answer.

    A model inside its skip window for this use is passed over at once; a timeout, 429 or 5xx opens its window
    (failures_before_skip, skip_scope). Unparsable structured output goes to the next model without counting. Inside a
    chat turn (T13d) a model that timed out earlier in the turn is passed over too, every deadline is cut to the time
    the turn has left, and a model is not started with less than MIN_ATTEMPT_S left. Raises LLMError when every
    candidate failed or was skipped (with none configured, when called).
    """

    def call(messages, config: RunnableConfig) -> Any:
        skip_list = skips or SKIPS
        if not candidates:
            raise LLMError("no LLM configured: set GOOGLE_API_KEY and/or GROQ_API_KEY + FALLBACK_MODEL")
        first_error: Exception | None = None
        turn = current_turn()
        for c in candidates:
            if turn is not None and c.key in turn.timed_out:
                first_error = _pass_over(c, "timed out earlier in this turn", None, first_error)
                continue
            left = skip_list.left(c.key, c.use)
            if left:
                first_error = _pass_over(c, "paused", left, first_error)
                continue
            if turn is not None and turn.left() < MIN_ATTEMPT_S:
                first_error = _pass_over(c, "no time left in this turn", None, first_error)
                continue
            try:
                return _run(c, messages, config, skip_list, c.seconds if turn is None else min(c.seconds, turn.left()))
            except Exception as exc:
                first_error = first_error or exc
        raise LLMError(f"every model failed or was skipped ({type(first_error).__name__})") from first_error

    use = candidates[0].use if candidates else "none"
    return RunnableLambda(call, name=f"model_chain:{use}")


def last_try_chain(candidates: list[Candidate], patient_main: Candidate | None, skips: SkipList | None = None
                   ) -> Runnable:
    """One last attempt after the answer chain failed (T13d), only inside a chat turn with at least LAST_TRY_MIN_S
    left; the agent calls it when no tool wrote anything this turn:
    (a) a model whose pause ends within LAST_TRY_WAIT_MAX_S (and still fits the turn): wait for it, try it once;
    (b) otherwise the main model once more with a long deadline (`patient_main`: LAST_TRY_S, a 25 s HTTP timeout),
        unless a 429 (its quota) is why it failed or is paused.
    In the Render logs of 13 Oct the 12:02:48 turn failed in 1.35 s while Groq's pause had only 6 s left. Raises
    LLMError when neither applies or the attempt fails."""

    def call(messages, config: RunnableConfig) -> Any:
        skip_list = skips or SKIPS
        turn = current_turn()
        if turn is None or turn.left() < LAST_TRY_MIN_S:
            raise LLMError("no time for a last try")
        waits = [(skip_list.left(c.key, c.use), c) for c in candidates if c.key not in turn.timed_out]
        waits = [(w, c) for w, c in waits if 0 < w <= LAST_TRY_WAIT_MAX_S and w + MIN_ATTEMPT_S <= turn.left()]
        if waits:
            wait, c = min(waits, key=lambda pair: pair[0])
            logger.info("llm last try", extra={"model": c.key, "use": c.use, "wait_s": round(wait, 1)})
            skip_list.sleep(wait)
            return _run(c, messages, config, skip_list, min(c.seconds, turn.left()))
        main = patient_main
        if main is None or "429" in (turn.failures.get(main.key), skip_list.paused_by(main.key, main.use)):
            raise LLMError("no model for a last try")
        seconds = min(main.seconds, turn.left())
        logger.info("llm last try", extra={"model": main.key, "use": main.use, "deadline_s": round(seconds, 1)})
        return _run(main, messages, config, skip_list, seconds)

    return RunnableLambda(call, name="last_try")


def _gemini(model: str, settings: Settings, use: str, seconds: float, tools=None, schema=None,
            http_timeout_s: float = GEMINI_MIN_TIMEOUT_S) -> Candidate:
    chat = gemini_chat(model, settings, http_timeout_s=http_timeout_s)
    return Candidate(f"gemini:{model}", _shape(chat, "gemini", tools, schema), use, seconds)


def _groq(settings: Settings, use: str, seconds: float, tools=None, schema=None, name: str | None = None) -> Candidate:
    """A Groq candidate; `name` picks FALLBACK_MODEL_2 instead of FALLBACK_MODEL. Each has its own skip-list key."""
    name = name or settings.fallback_model
    model = _shape(groq_chat(settings, seconds, name), "groq", tools, schema)
    return Candidate(f"groq:{name}", model, use, seconds)


def _build(make: Callable[[], list[Candidate]], what: str) -> list[Candidate]:
    """The candidates, or none if building them fails (the chain then raises LLMError when called)."""
    try:
        return make()
    except Exception as exc:
        logger.error("building the %s models failed: %s", what, type(exc).__name__)
        return []


def answer_candidates(
    model: str, settings: Settings | None = None, *, tools: list | None = None, schema: type | None = None
) -> list[Candidate]:
    """The answer step's models: Gemini `model` (8 s, one attempt), then Groq (10 s), then (T13d) the second Groq model
    (10 s, FALLBACK_MODEL_2: its own tokens a minute, so a 429 on the first doesn't end the chain), then the fast Gemini
    model (8 s) as the last resort. Unconfigured ones are left out. tools: all get them bound (Groq without parallel
    calls); schema: all return {"raw", "parsed", "parsing_error"}.

    Why the fast model last (T12a live run): with the main model timing out and Groq out of tokens for the
    minute (its 8K tokens a minute carry about one turn), every turn ended in "try again"; the fast model, on
    its own quota, was answering in about 2 s. It is reached only when both others fail or are skipped.
    """
    settings = settings or get_settings()

    def make() -> list[Candidate]:
        candidates = []
        if settings.google_api_key and model:
            candidates.append(_gemini(model, settings, "answer", ANSWER_TIMEOUT_S, tools, schema))
        if settings.groq_api_key and settings.fallback_model:
            candidates.append(_groq(settings, "answer", GROQ_ANSWER_TIMEOUT_S, tools, schema))
        second = settings.fallback_model_2
        if settings.groq_api_key and second and second != settings.fallback_model:
            candidates.append(_groq(settings, "answer", GROQ_ANSWER_TIMEOUT_S, tools, schema, second))
        fast = settings.llm_model_fast
        if settings.google_api_key and fast and fast != model:
            candidates.append(_gemini(fast, settings, "answer", ANSWER_TIMEOUT_S, tools, schema))
        return candidates

    return _build(make, "answer")


def chat_llm(
    model: str,
    settings: Settings | None = None,
    *,
    tools: list | None = None,
    schema: type | None = None,
) -> Runnable:
    """Gemini `model`, then Groq, then the fast model (answer_candidates), through model_chain. Never raises when
    built; with no provider configured, it raises LLMError when called, so the chat replies "degraded" instead of
    crashing. Worst case, when all three hang before any skip window is open: 8 + 10 + 8 = 26 s (it was about
    46 s before T12a for two models); once a window is open, a skipped model costs nothing.

    T13d: the chain carries `last_try` (last_try_chain over the same models plus the main model with a 20 s deadline),
    which the agent calls once when the chain failed and no tool wrote anything this turn."""
    settings = settings or get_settings()
    candidates = answer_candidates(model, settings, tools=tools, schema=schema)
    chain = model_chain(candidates)
    long_main = _build(lambda: [_gemini(model, settings, "answer", LAST_TRY_S, tools, schema, LAST_TRY_HTTP_S)]
                       if settings.google_api_key and model else [], "last-try")
    chain.last_try = last_try_chain(candidates, long_main[0] if long_main else None)
    return chain


@lru_cache
def main_llm() -> Runnable:
    """The model that writes answers (LLM_MODEL_MAIN), without tools."""
    return chat_llm(get_settings().llm_model_main)


def reason_candidates(settings: Settings | None = None, seconds: float = REASON_TIMEOUT_S) -> list[Candidate]:
    """The lead reason's model (T13d): the fast model only, 8 s, so it never uses the main model's quota, which the
    customers' turns need. It runs after the reply (or offline), so its slower days cost nobody a wait; the caller
    uses the template when it fails. Groq is left out to keep its tokens for answers. No fast model: no candidates."""
    settings = settings or get_settings()

    def make() -> list[Candidate]:
        if not (settings.google_api_key and settings.llm_model_fast):
            return []
        return [_gemini(settings.llm_model_fast, settings, "reason", seconds)]

    return _build(make, "reason")


@lru_cache
def reason_llm() -> Runnable:
    """The model for the lead reason; write_reason uses the template sentence when it fails or is skipped."""
    return model_chain(reason_candidates())


def understand_candidates(
    schema: type, settings: Settings | None = None, seconds: float = UNDERSTAND_TIMEOUT_S
) -> list[Candidate]:
    """The understand step's models in order (T13d): main Gemini, then fast Gemini, each one attempt cut off after
    4 s, with structured output and the raw message kept. Groq is kept for answers: its 8K tokens a minute carry about
    one turn, and an understand call used a third of it (Render logs of 13 Oct: Groq's 429 came on an answer after
    two understand calls). When both fail, the understand node uses the raw message and reads details in code.
    Unconfigured ones are left out."""
    settings = settings or get_settings()

    def make() -> list[Candidate]:
        candidates = []
        if settings.google_api_key and settings.llm_model_main:
            candidates.append(_gemini(settings.llm_model_main, settings, "understand", seconds, schema=schema))
        if settings.google_api_key and settings.llm_model_fast and settings.llm_model_fast != settings.llm_model_main:
            candidates.append(_gemini(settings.llm_model_fast, settings, "understand", seconds, schema=schema))
        return candidates

    return _build(make, "understand")


def understand_llm(schema: type, settings: Settings | None = None) -> Runnable:
    """The understand step's model: understand_candidates through model_chain. Raises when every one fails
    or is skipped; the understand node then searches with the raw message."""
    return model_chain(understand_candidates(schema, settings))


def offline_understand_llm(schema: type, settings: Settings | None = None) -> Runnable:
    """The understand step for offline scripts (eval_retrieval --rewrite, T12b): the same models in the same order,
    30 s each, no skip list, the chain retried after a timeout, 429 or 5xx."""
    candidates = understand_candidates(schema, settings, seconds=OFFLINE_TIMEOUT_S)
    return patient(model_chain(candidates, skips=NoSkips()))


def offline_reason_llm(settings: Settings | None = None) -> Runnable:
    """The lead reason for offline scripts (seed_demo_leads, T12b): the fast model (T13d), 30 s, retried like
    above."""
    return patient(model_chain(reason_candidates(settings, seconds=OFFLINE_TIMEOUT_S), skips=NoSkips()))


def provider_of(message: BaseMessage) -> str:
    """Which provider wrote a message: gemini, groq or unknown."""
    return PROVIDERS.get(message.response_metadata.get("model_provider", ""), "unknown")


def reply_text(message: BaseMessage) -> str:
    """Only the answer text: text blocks only, and any inline <think>...</think> removed.

    Reasoning in additional_kwargs (Groq) or thinking blocks (Gemini) is never read.
    """
    return _THINK_RE.sub("", message.text).strip()


def invoke_llm(llm: Runnable, messages: list[BaseMessage]) -> LLMReply:
    """Call a chat model, time it and log which provider answered. Raises LLMError on any failure."""
    start = time.perf_counter()
    try:
        message = llm.invoke(messages)
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000)
        logger.warning(
            "llm call failed: %s", type(exc).__name__, extra={"latency_ms": latency_ms}
        )
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc
    latency_ms = round((time.perf_counter() - start) * 1000)
    provider = provider_of(message)
    logger.info(
        "llm call",
        extra={"provider": provider, "latency_ms": latency_ms, "fallback_used": provider == "groq"},
    )
    text = reply_text(message)
    if not text:
        raise LLMError(f"empty reply from {provider}")
    return LLMReply(text=text, provider=provider, latency_ms=latency_ms)

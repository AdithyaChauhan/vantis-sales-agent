"""LLM access: the embedder, plus main_llm() and fast_llm() chat models. Everything else uses these.

Embedder: google-genai directly. gemini-embedding-2 has no task_type, so the task goes in the text:
documents are "title: ... | text: ...", queries are "task: question answering | query: ...".

Chat models: Gemini (langchain-google-genai) with an automatic fallback to Groq (langchain-groq).
The understand step uses understand_llm(): several models in turn, each cut off after 5 s.
"""
import contextvars
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

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
# Chat path: fail fast so search can fall back to keyword (worst case about 22 s).
QUERY_HTTP = types.HttpOptions(
    timeout=10_000,
    retry_options=types.HttpRetryOptions(
        attempts=2, initial_delay=1, max_delay=2, http_status_codes=RETRY_STATUS_CODES
    ),
)


class EmbeddingError(RuntimeError):
    """Embedding failed after retries, or the response had the wrong shape."""


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

    def embed_query(self, text: str) -> list[float]:
        """Embed one question (cached, so a repeated question costs no quota)."""
        return list(self._query_cache(text))

    def _embed_query_uncached(self, text: str) -> tuple[float, ...]:
        return tuple(self._embed([query_text(text)], QUERY_HTTP)[0])

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

LLM_TIMEOUT_S = 15
# Gemini: 15 s per attempt, 2 attempts (one retry) on timeouts, 408 and 5xx. 429 is left out on
# purpose: a quota error will not clear in a second, so it goes straight to Groq.
# Worst case for one reply: 15 s + ~1 s + 15 s on Gemini, then 15 s on Groq (about 46 s).
GEMINI_CHAT_HTTP = types.HttpOptions(
    timeout=LLM_TIMEOUT_S * 1000,
    retry_options=types.HttpRetryOptions(
        attempts=2, initial_delay=1, max_delay=2, http_status_codes=[408, 500, 502, 503, 504]
    ),
)
# The understand step (T10b): one attempt per model, cut off after 5 s, then the next model answers.
# Measured on the same understand input (3 calls each): main 1.7 s median (worst 1.9 s), Groq 1.0 s (but
# about 2,300 tokens a call against its 8K tokens a minute), fast 17.4 s (worst 18.5 s). So main goes first.
UNDERSTAND_TIMEOUT_S = 5
# The Gemini API refuses a deadline under 10 s (400 "Minimum allowed deadline is 10s"), so the HTTP timeout
# is 10 s with no retry, and with_deadline cuts the call off at 5 s.
GEMINI_MIN_TIMEOUT_S = 10
UNDERSTAND_HTTP = types.HttpOptions(
    timeout=GEMINI_MIN_TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)
)
# A late call keeps its worker busy until its own HTTP timeout (Gemini 10 s, Groq 5 s), so a worst-case
# turn holds up to 3 workers for up to 10 s. 8 workers keep calls out of the queue (the deadline counts
# from submit) at our traffic: about 5 turns a minute, the main model's limit.
DEADLINE_WORKERS = 8
QUEUE_WARN_S = 0.5
THINKING_LEVEL = "minimal"  # Gemini 3.x thinks by default; minimal is the lowest level (fastest)
GROQ_MAX_RETRIES = 0
GROQ_REASONING_EFFORT = "low"  # keeps the fallback fast and under Groq's 8K tokens/min
PROVIDERS = {"google_genai": "gemini", "groq": "groq"}  # response_metadata["model_provider"] -> name
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """The chat model failed (after retries and the fallback), or gave an empty reply."""


@dataclass
class LLMReply:
    """The answer text, which provider wrote it (gemini | groq | unknown) and how long it took."""

    text: str
    provider: str
    latency_ms: int


class GeminiChat(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI that sends our timeout and retry rules (request_http, by default
    GEMINI_CHAT_HTTP) with every request.

    bind_tools() drops bound http_options and with_structured_output() rejects them, so they are added
    here, where every call (plain, tools, structured output) builds its request. This relies on a private
    method of langchain-google-genai 4.4.1 (pinned; tests/test_llm.py checks it).
    """

    request_http: Any = GEMINI_CHAT_HTTP

    def _prepare_request(self, messages, **kwargs):
        kwargs.setdefault("http_options", self.request_http)
        return super()._prepare_request(messages, **kwargs)


def gemini_chat(model: str, settings: Settings, http: types.HttpOptions = GEMINI_CHAT_HTTP) -> GeminiChat:
    """Gemini chat model: minimal thinking, our timeout and retry rules (or `http`)."""
    return GeminiChat(
        model=model, google_api_key=settings.google_api_key, thinking_level=THINKING_LEVEL, request_http=http
    )


def groq_chat(settings: Settings, timeout_s: float = LLM_TIMEOUT_S) -> ChatGroq:
    """Groq fallback model (FALLBACK_MODEL). One attempt; its reasoning never reaches the reply."""
    return ChatGroq(
        model=settings.fallback_model,
        api_key=settings.groq_api_key,
        timeout=timeout_s,
        max_retries=GROQ_MAX_RETRIES,
        reasoning_effort=GROQ_REASONING_EFFORT,
    )


def _not_configured(_messages) -> None:
    raise LLMError("no LLM configured: set GOOGLE_API_KEY and/or GROQ_API_KEY + FALLBACK_MODEL")


def _shape(model: Runnable, provider: str, tools: list | None, schema: type | None) -> Runnable:
    """The model as plain chat, with tools bound, or with structured output (raw message kept)."""
    if tools:
        # gpt-oss on Groq does not support parallel tool calls
        extra = {"parallel_tool_calls": False} if provider == "groq" else {}
        return model.bind_tools(tools, **extra)
    if schema is not None:
        return model.with_structured_output(schema, include_raw=True)
    return model


def chat_llm(
    model: str,
    settings: Settings | None = None,
    *,
    tools: list | None = None,
    schema: type | None = None,
) -> Runnable:
    """Gemini `model`, falling back to Groq on any error. Never raises.

    tools: both models get the tools bound (each built separately, as fallbacks need).
    schema: both return {"raw": AIMessage, "parsed": schema | None, "parsing_error": ...}.
    With only one provider configured, that one is used alone. With none (or if building fails),
    the result raises LLMError when called, so the chat replies "degraded" instead of crashing.
    """
    settings = settings or get_settings()
    try:
        gemini = gemini_chat(model, settings) if settings.google_api_key and model else None
        groq = groq_chat(settings) if settings.groq_api_key and settings.fallback_model else None
        gemini = _shape(gemini, "gemini", tools, schema) if gemini else None
        groq = _shape(groq, "groq", tools, schema) if groq else None
    except Exception as exc:
        logger.error("building the chat model failed: %s", type(exc).__name__)
        gemini = groq = None
    if gemini and groq:
        return gemini.with_fallbacks([groq])
    return gemini or groq or RunnableLambda(_not_configured)


@lru_cache
def main_llm() -> Runnable:
    """The model that writes answers (LLM_MODEL_MAIN). Also the FastAPI dependency tests override."""
    return chat_llm(get_settings().llm_model_main)


@lru_cache
def fast_llm() -> Runnable:
    """The model for small steps (LLM_MODEL_FAST), e.g. the lead reason."""
    return chat_llm(get_settings().llm_model_fast)


# ---------- the understand step: short deadlines, then the next model ----------

_deadline_pool = ThreadPoolExecutor(max_workers=DEADLINE_WORKERS, thread_name_prefix="llm-deadline")


def with_deadline(model: Runnable, seconds: float = UNDERSTAND_TIMEOUT_S) -> Runnable:
    """`model`, but raising LLMError if it has not answered within `seconds` (wall clock) or its structured
    output could not be parsed, so a fallback answers instead.

    The HTTP timeout cannot do this: Gemini refuses deadlines under 10 s, and httpx times each network
    phase, not the whole call. This guard can also be tested with a fake slow model. The call runs on a
    small thread pool; a late call is left to finish there and its own HTTP timeout ends it.
    """

    def call(messages, config: RunnableConfig) -> Any:
        queued = time.perf_counter()

        def run():
            waited = time.perf_counter() - queued
            if waited > QUEUE_WARN_S:  # the deadline counts from here too: the pool is too small
                logger.warning("llm call waited for a worker", extra={"latency_ms": round(waited * 1000)})
            return model.invoke(messages, config)

        future = _deadline_pool.submit(contextvars.copy_context().run, run)  # keeps tracing context
        try:
            result = future.result(timeout=seconds)
        except FutureTimeout:
            logger.warning("llm call cut off", extra={"latency_ms": round(seconds * 1000)})
            raise LLMError(f"no answer within {seconds} s") from None
        if isinstance(result, dict) and "parsed" in result and result["parsed"] is None:
            raise LLMError("structured output could not be parsed")
        return result

    return RunnableLambda(call, name="with_deadline")


def deadline_chain(models: list[Runnable], seconds: float = UNDERSTAND_TIMEOUT_S) -> Runnable:
    """Every model gets its own `seconds`; on a timeout or error the next one answers. When all fail it raises
    the first one's error (with no models, LLMError when called); understand() turns that into LLMError."""
    wrapped = [with_deadline(m, seconds) for m in models]
    if not wrapped:
        return RunnableLambda(_not_configured)
    return wrapped[0].with_fallbacks(wrapped[1:]) if len(wrapped) > 1 else wrapped[0]


def understand_candidates(schema: type, settings: Settings | None = None) -> list[Runnable]:
    """The understand step's models in order (T10b measurement): main Gemini, fast Gemini, then Groq.
    One attempt each (HTTP timeout: Gemini 10 s, its minimum; Groq 5 s), structured output with the raw
    message kept; understand_llm cuts each off after 5 s. Groq is last: its 8K tokens a minute is shared
    with the agent's own fallback. Unconfigured ones are left out."""
    settings = settings or get_settings()
    candidates = []
    try:
        if settings.google_api_key:
            for model in (settings.llm_model_main, settings.llm_model_fast):
                if model:
                    candidates.append(_shape(gemini_chat(model, settings, UNDERSTAND_HTTP), "gemini", None, schema))
        if settings.groq_api_key and settings.fallback_model:
            candidates.append(_shape(groq_chat(settings, UNDERSTAND_TIMEOUT_S), "groq", None, schema))
    except Exception as exc:
        logger.error("building the understand models failed: %s", type(exc).__name__)
        return []
    return candidates


def understand_llm(schema: type, settings: Settings | None = None) -> Runnable:
    """The understand step's model: understand_candidates, each cut off after UNDERSTAND_TIMEOUT_S.
    Raises when every one fails; the understand node then searches with the raw message."""
    return deadline_chain(understand_candidates(schema, settings), UNDERSTAND_TIMEOUT_S)


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

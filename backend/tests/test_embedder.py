"""Embedder tests with a fake Gemini client. No network."""
import time
from types import SimpleNamespace

import pytest

from app.core import llm
from app.core.llm import DOCUMENT_HTTP, QUERY_HTTP, Embedder, EmbeddingError, query_text
from app.rag.chunker import Chunk

DIM = 4


class FakeClient:
    """Records every embed_content call; returns one vector per Content it receives."""

    def __init__(self, dim=DIM, drop=0):
        self.models = self
        self.calls = []
        self.dim = dim
        self.drop = drop  # how many vectors to leave out of the response

    def embed_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        count = len(contents) - self.drop
        values = [float(len(self.calls))] * self.dim
        return SimpleNamespace(embeddings=[SimpleNamespace(values=list(values)) for _ in range(count)])


def make(client=None):
    client = client or FakeClient()
    return Embedder(client, "fake-model", DIM), client


def test_each_text_is_its_own_content():
    embedder, client = make()
    vectors = embedder.embed_documents(["one", "two", "three"])
    assert len(vectors) == 3
    contents = client.calls[0].contents
    assert len(contents) == 3
    assert [c.parts[0].text for c in contents] == ["one", "two", "three"]
    assert all(len(c.parts) == 1 for c in contents)
    assert client.calls[0].config.output_dimensionality == DIM
    assert client.calls[0].model == "fake-model"


def test_batches_of_20():
    embedder, client = make()
    vectors = embedder.embed_documents([f"t{i}" for i in range(25)])
    assert len(vectors) == 25
    assert [len(c.contents) for c in client.calls] == [20, 5]


def test_empty_input_makes_no_call():
    embedder, client = make()
    assert embedder.embed_documents([]) == []
    assert client.calls == []


def test_count_mismatch_raises():
    embedder, _ = make(FakeClient(drop=1))
    with pytest.raises(EmbeddingError, match="got 1"):
        embedder.embed_documents(["a", "b"])


def test_wrong_dimension_raises():
    embedder, _ = make(FakeClient(dim=3))
    with pytest.raises(EmbeddingError, match="expected 4"):
        embedder.embed_documents(["a"])


def test_api_error_becomes_embedding_error():
    class Broken(FakeClient):
        def embed_content(self, **kwargs):
            raise TimeoutError("slow")

    embedder, _ = make(Broken())
    with pytest.raises(EmbeddingError, match="TimeoutError"):
        embedder.embed_query("hello")


def test_query_has_task_prefix_and_is_cached():
    embedder, client = make()
    first = embedder.embed_query("Is there a guarantee?")
    second = embedder.embed_query("Is there a guarantee?")
    assert first == second
    assert len(client.calls) == 1  # the repeat came from the cache
    assert client.calls[0].contents[0].parts[0].text == query_text("Is there a guarantee?")
    assert query_text("x") == "task: question answering | query: x"
    embedder.embed_query("another question")
    assert len(client.calls) == 2


def test_cache_returns_a_copy():
    embedder, _ = make()
    vector = embedder.embed_query("q")
    vector.append(99.0)
    assert len(embedder.embed_query("q")) == DIM


def test_query_fails_fast_and_documents_are_patient():
    embedder, client = make()
    embedder.embed_query("q")
    embedder.embed_documents(["d"])
    query_options = client.calls[0].config.http_options
    document_options = client.calls[1].config.http_options
    # T12a: one query attempt; the 3 s wall-clock deadline ends it (Gemini refuses HTTP deadlines under 10 s)
    assert (query_options.timeout, query_options.retry_options.attempts) == (10_000, 1)
    assert (document_options.timeout, document_options.retry_options.attempts) == (30_000, 3)
    assert query_options is QUERY_HTTP and document_options is DOCUMENT_HTTP
    assert llm.QUERY_TIMEOUT_S == 3


class SlowClient(FakeClient):
    """Answers each call after `delay` seconds."""

    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def embed_content(self, **kwargs):
        time.sleep(self.delay)
        return super().embed_content(**kwargs)


class ScriptedClient(FakeClient):
    """Fails ("fail": a 503) or answers ("ok"), call by call."""

    def __init__(self, *steps):
        super().__init__()
        self.steps = list(steps)

    def embed_content(self, **kwargs):
        step = self.steps[min(len(self.calls), len(self.steps) - 1)]
        if step == "fail":
            self.calls.append(None)
            error = RuntimeError("503 UNAVAILABLE")
            error.code = 503
            raise error
        return super().embed_content(**kwargs)


def test_a_slow_query_embedding_is_cut_off(monkeypatch):
    monkeypatch.setattr(llm, "QUERY_TIMEOUT_S", 0.2)  # the real value is 3 s (checked above)
    embedder, _ = make(SlowClient(delay=1))
    start = time.perf_counter()
    with pytest.raises(EmbeddingError):  # search() then falls back to keyword-only
        embedder.embed_query("slow question")
    assert time.perf_counter() - start < 0.6


def test_one_failure_does_not_skip_the_embedder_but_two_in_a_row_do():
    embedder, client = make(ScriptedClient("fail", "fail", "ok"))
    key = "gemini-embed:fake-model"
    with pytest.raises(EmbeddingError):
        embedder.embed_query("first")
    assert llm.SKIPS.left(key, "embed") == 0  # one blip costs one fallback, not the whole window
    with pytest.raises(EmbeddingError):
        embedder.embed_query("second")
    assert llm.SKIPS.left(key, "embed") > 0  # two in a row: skipped for a while
    with pytest.raises(EmbeddingError, match="skipped"):
        embedder.embed_query("third")
    assert len(client.calls) == 2  # the skipped query never reached the API


def test_a_success_resets_the_embedders_count():
    embedder, _ = make(ScriptedClient("fail", "ok", "fail"))
    for text in ("one", "two", "three"):
        try:
            embedder.embed_query(text)
        except EmbeddingError:
            pass
    assert llm.SKIPS.left("gemini-embed:fake-model", "embed") == 0


def test_embedder_without_key_raises(monkeypatch):
    llm.embedder.cache_clear()
    monkeypatch.setattr(
        llm, "get_settings", lambda: SimpleNamespace(google_api_key="", embedding_model="m", embedding_dim=4)
    )
    with pytest.raises(EmbeddingError, match="GOOGLE_API_KEY"):
        llm.embedder()
    llm.embedder.cache_clear()


def test_document_text_format():
    chunk = Chunk(
        "s08-x", 8, "Product & Service Catalog", 3, "table_row", "Size: 5 kW",
        metadata={"label": "Vantis HomePro 5"},
    )
    assert chunk.embed_text() == (
        "title: KB Section 8: Product & Service Catalog - Vantis HomePro 5 | text: Size: 5 kW"
    )
    prose = Chunk("s05-pricing", 5, "Pricing", 2, "prose", "Prices vary.")
    assert prose.embed_text() == "title: KB Section 5: Pricing | text: Prices vary."
    piped = Chunk("s15-q", 15, "FAQ", 5, "faq", "A", metadata={"label": "Cost | price?"})
    assert "|" not in piped.embed_title()


# ---------- offline scripts (T12b): patient query embeddings ----------

def test_embed_queries_batches_query_texts_with_the_patient_options():
    embedder, client = make()
    vectors = embedder.embed_queries([f"q{i}" for i in range(22)])
    assert len(vectors) == 22
    assert [len(c.contents) for c in client.calls] == [20, 2]
    assert client.calls[0].contents[0].parts[0].text == query_text("q0")
    assert client.calls[0].config.http_options is DOCUMENT_HTTP  # 30 s and 3 attempts on 429/5xx, like ingest


def test_embed_queries_ignores_the_chat_skip_list(monkeypatch):
    embedder, client = make()
    monkeypatch.setattr(llm.SKIPS, "left", lambda key: 60.0)  # the chat path would refuse now
    assert len(embedder.embed_queries(["q"])) == 1

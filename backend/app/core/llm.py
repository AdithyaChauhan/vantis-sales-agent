"""LLM access. For now only the embedder (T5 adds the chat models).

Uses google-genai directly. gemini-embedding-2 has no task_type, so the task goes in the text:
documents are "title: ... | text: ...", queries are "task: question answering | query: ...".
"""
import logging
from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import get_settings

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

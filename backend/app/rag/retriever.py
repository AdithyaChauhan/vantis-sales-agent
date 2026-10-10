"""Hybrid retrieval over kb_chunks: keyword (Postgres full-text) + vector (pgvector), fused with RRF.

Policy chunks (KB Sections 16-19) are instructions for the assistant, so they are never returned.
"""
import logging
from dataclasses import dataclass, field, replace

from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine, bindparam, text

from app.core.config import get_settings
from app.core.db import get_engine
from app.core.llm import embedder as get_embedder

logger = logging.getLogger(__name__)

MODES = ("hybrid", "keyword", "vector")
CANDIDATES = 10  # how many hits each search contributes before fusion
RRF_K = 60

_COLUMNS = "chunk_id, section_no, section_title, page, chunk_type, content"

# plainto_tsquery ANDs every word ("Do you provide installation in Delhi?" would match nothing),
# so the ' & ' between terms is replaced by ' | '. Stop words are dropped by the 'english' config.
_KEYWORD_SQL = text(
    f"""
    WITH q AS (
        SELECT CAST(replace(plainto_tsquery('english', :q)::text, ' & ', ' | ') AS tsquery) AS tsq
    )
    SELECT {_COLUMNS}, ts_rank_cd(tsv, tsq) AS score
    FROM kb_chunks, q
    WHERE tsv @@ tsq AND chunk_type <> 'policy'
    ORDER BY score DESC, chunk_id
    LIMIT :limit
    """
)

_VECTOR_SQL = text(
    f"""
    SELECT {_COLUMNS}, embedding <=> CAST(:qvec AS vector) AS distance
    FROM kb_chunks
    WHERE chunk_type <> 'policy'
    ORDER BY distance, chunk_id
    LIMIT :limit
    """
).bindparams(bindparam("qvec", type_=Vector()))


@dataclass
class RetrievedChunk:
    """A chunk returned by search, with where it ranked in each search."""

    chunk_id: str
    section_no: int
    section_title: str
    page: int
    chunk_type: str
    content: str
    keyword_rank: int | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    similarity: float | None = None  # cosine similarity, 1 - distance
    rrf_score: float | None = None


@dataclass
class SearchResult:
    """What search() returns. `mode` is the mode actually used (keyword after a fallback)."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    mode: str = "hybrid"
    no_context: bool = False
    degraded: bool = False


def rrf_fuse(
    keyword: list[RetrievedChunk], vector: list[RetrievedChunk], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over the lists a chunk appears in.

    Ranks start at 1 (the position in each list). Inputs are not changed.
    Ties are broken by chunk_id so the order is stable.
    """
    merged: dict[str, RetrievedChunk] = {}
    for rank, hit in enumerate(keyword, start=1):
        entry = merged.setdefault(hit.chunk_id, replace(hit, rrf_score=0.0))
        entry.keyword_rank, entry.keyword_score = rank, hit.keyword_score
        entry.rrf_score += 1 / (k + rank)
    for rank, hit in enumerate(vector, start=1):
        entry = merged.setdefault(hit.chunk_id, replace(hit, rrf_score=0.0))
        entry.vector_rank, entry.similarity = rank, hit.similarity
        entry.rrf_score += 1 / (k + rank)
    return sorted(merged.values(), key=lambda c: (-c.rrf_score, c.chunk_id))


def has_context(best_similarity: float | None, keyword_hits: int, min_similarity: float) -> bool:
    """Relevance gate: there is context if any keyword matched or the best similarity is high enough."""
    if keyword_hits > 0:
        return True
    return best_similarity is not None and best_similarity >= min_similarity


def keyword_search(engine: Engine, query: str, limit: int = CANDIDATES) -> list[RetrievedChunk]:
    """Top chunks by ts_rank_cd, matching ANY of the query words."""
    with engine.connect() as conn:
        rows = conn.execute(_KEYWORD_SQL, {"q": query, "limit": limit}).mappings().all()
    return [
        RetrievedChunk(**_chunk_fields(row), keyword_rank=rank, keyword_score=float(row["score"]))
        for rank, row in enumerate(rows, start=1)
    ]


def vector_search(
    engine: Engine, vector: list[float], limit: int = CANDIDATES
) -> list[RetrievedChunk]:
    """Top chunks by cosine distance to the query vector."""
    with engine.connect() as conn:
        rows = conn.execute(_VECTOR_SQL, {"qvec": vector, "limit": limit}).mappings().all()
    return [
        RetrievedChunk(
            **_chunk_fields(row), vector_rank=rank, similarity=1 - float(row["distance"])
        )
        for rank, row in enumerate(rows, start=1)
    ]


def _chunk_fields(row) -> dict:
    return {
        "chunk_id": row["chunk_id"],
        "section_no": row["section_no"],
        "section_title": row["section_title"],
        "page": row["page"],
        "chunk_type": row["chunk_type"],
        "content": row["content"],
    }


def search(
    query: str,
    k: int = 4,
    mode: str = "hybrid",
    *,
    engine: Engine | None = None,
    embedder=None,
    min_similarity: float | None = None,
) -> SearchResult:
    """Find the top k KB chunks for a question.

    If embedding the query fails, falls back to keyword-only and sets degraded=True (no raise).
    Chunks are returned even when no_context is True, so callers can see the near misses;
    the answering code must check no_context. Database errors are not caught here.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    query = " ".join(query.split())
    if not query:
        return SearchResult(mode=mode, no_context=True)
    if engine is None:
        engine = get_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is not set; cannot search the KB")
    threshold = get_settings().retrieval_min_similarity if min_similarity is None else min_similarity

    vector_hits: list[RetrievedChunk] = []
    degraded = False
    if mode in ("hybrid", "vector"):
        try:
            embedder = embedder if embedder is not None else get_embedder()
            vector_hits = vector_search(engine, embedder.embed_query(query))
        except Exception as exc:  # embedding or vector search failed: keep chatting on keywords
            logger.warning("vector search failed, using keyword only: %s", type(exc).__name__)
            degraded, mode = True, "keyword"

    keyword_hits = keyword_search(engine, query) if mode in ("hybrid", "keyword") else []

    if mode == "hybrid":
        ranked = rrf_fuse(keyword_hits, vector_hits)
    elif mode == "vector":
        ranked = vector_hits
    else:
        ranked = keyword_hits
    best_similarity = vector_hits[0].similarity if vector_hits else None
    return SearchResult(
        chunks=ranked[:k],
        mode=mode,
        no_context=not has_context(best_similarity, len(keyword_hits), threshold),
        degraded=degraded,
    )


def count_kb_chunks(engine: Engine | None) -> int | None:
    """Number of rows in kb_chunks, or None if the table is missing or the DB is unreachable."""
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            return int(conn.execute(text("SELECT count(*) FROM kb_chunks")).scalar_one())
    except Exception as exc:
        logger.warning("kb_chunks count failed: %s", type(exc).__name__)
        return None

"""Database tests for ingest, retrieval and service areas.

They run only against TEST_DATABASE_URL (the `test_engine` fixture) and use fake one-hot vectors,
so they never touch the main database and never call the embedding API.
"""
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.llm import EmbeddingError
from app.rag.chunker import chunk_sections
from app.rag.ingest import ingest_sections, write_kb
from app.rag.models import KbChunk, ServiceArea
from app.rag.pdf_parser import Line, Section, Table
from app.rag.retriever import keyword_search, search, vector_search, word_doc_counts
from app.services.service_areas import ServiceAreas

DIM = get_settings().embedding_dim

SECTIONS = [
    Section(3, "Solar Solutions", 2, lines=[Line("Rooftop solar panels for homes.", 2, True)]),
    Section(
        10, "Reference Service Locations", 3,
        tables=[
            Table(3, ["Region", "Reference Locations"],
                  [["Uttar Pradesh", "Lucknow, Noida"], ["Delhi NCR", "Delhi, Noida"]])
        ],
    ),
    Section(11, "Warranty and Support", 4, lines=[Line("Panels carry a guarantee against defects.", 4, True)]),
    Section(16, "Must Not Invent", 6, lines=[Line("Never invent a guarantee or a price.", 6, True)]),
]
CHUNKS = chunk_sections(SECTIONS)
INDEX = {c.chunk_id: i for i, c in enumerate(CHUNKS)}
SOLAR, WARRANTY, POLICY = "s03-solar-solutions", "s11-warranty-and-support", "s16-must-not-invent"


def mix(*weights: tuple[str, float]) -> list[float]:
    """A vector with the given weight on each named chunk's axis."""
    vector = [0.0] * DIM
    for chunk_id, weight in weights:
        vector[INDEX[chunk_id]] = weight
    return vector


class FakeEmbedder:
    """Chunk number i gets the unit vector on axis i. The query vector is set by the test."""

    model = "fake-embed"
    dim = DIM

    def __init__(self):
        self.query_vector = mix((SOLAR, 1.0))

    def embed_documents(self, texts):
        return [mix((c.chunk_id, 1.0)) for c in CHUNKS][: len(texts)]

    def embed_query(self, text):
        return self.query_vector


class BrokenEmbedder:
    model = "broken"
    dim = DIM

    def embed_documents(self, texts):
        raise EmbeddingError("quota exceeded")


@pytest.fixture
def kb(test_engine):
    """A freshly ingested synthetic KB in the test database."""
    embedder = FakeEmbedder()
    result = ingest_sections(SECTIONS, "abc123def456", test_engine, embedder)
    return SimpleNamespace(engine=test_engine, embedder=embedder, result=result)


def snapshot(engine):
    with Session(engine) as session:
        return sorted(session.execute(select(KbChunk.chunk_id, KbChunk.content, KbChunk.doc_version)).all())


def count(engine, model):
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(model))


def test_ingest_writes_chunks_and_service_areas(kb):
    assert count(kb.engine, KbChunk) == len(CHUNKS) == 6  # incl. the Section 10 summary chunk
    assert count(kb.engine, ServiceArea) == 4  # Lucknow, Noida x2, Delhi
    with Session(kb.engine) as session:
        row = session.get(KbChunk, WARRANTY)
        assert row.doc_version == "abc123def456"
        assert row.meta["embedding_model"] == "fake-embed"
        assert len(row.embedding) == DIM
        assert session.scalar(select(KbChunk.tsv).where(KbChunk.chunk_id == WARRANTY))  # generated column


def test_ingest_twice_gives_the_same_rows(kb):
    before = snapshot(kb.engine)
    ingest_sections(SECTIONS, "abc123def456", kb.engine, FakeEmbedder())
    assert snapshot(kb.engine) == before
    assert count(kb.engine, ServiceArea) == 4


def test_failed_embedding_leaves_the_database_unchanged(kb):
    before = snapshot(kb.engine)
    changed = [Section(3, "Solar Solutions", 2, lines=[Line("Totally different text.", 2, True)])]
    changed += SECTIONS[1:]
    with pytest.raises(EmbeddingError):
        ingest_sections(changed, "ffffffffffff", kb.engine, BrokenEmbedder())
    assert snapshot(kb.engine) == before
    assert count(kb.engine, ServiceArea) == 4


def test_failed_write_rolls_back_everything(kb):
    before = snapshot(kb.engine)
    duplicate_areas = [("Pune", "Maharashtra"), ("Pune", "Maharashtra")]  # breaks the unique constraint
    with pytest.raises(IntegrityError):
        write_kb(kb.engine, [], duplicate_areas)
    assert snapshot(kb.engine) == before  # the delete was rolled back too
    assert count(kb.engine, ServiceArea) == 4


def test_keyword_hit_matches_any_word(kb):
    # "zebra" and "stripes" match nothing; an AND query would return no rows at all.
    hits = keyword_search(kb.engine, "guarantee on zebra stripes")
    assert [h.chunk_id for h in hits] == [WARRANTY]
    assert hits[0].keyword_rank == 1 and hits[0].keyword_score > 0


def test_keyword_search_excludes_policy(kb):
    hits = keyword_search(kb.engine, "guarantee price")
    assert POLICY not in [h.chunk_id for h in hits]  # the policy text matches both words


def test_keyword_search_with_only_stop_words_returns_nothing(kb):
    assert keyword_search(kb.engine, "what is the") == []


def test_word_doc_counts_skip_the_policy_chunk(kb):
    counts, total = word_doc_counts(kb.engine)
    assert total == 5  # 6 chunks (incl. the Section 10 summary), one of them policy
    assert counts["panel"] == 2  # solar and warranty chunks
    assert counts["guarante"] == 1  # the policy chunk also says "guarantee" but is not counted
    assert "invent" not in counts  # only the policy chunk has this word


def test_common_words_are_dropped_from_the_keyword_query(kb):
    # "panel" is in 2 of the 5 chunks (40%, above the 30% cutoff), so only "guarantee" is searched
    assert [h.chunk_id for h in keyword_search(kb.engine, "panels guarantee")] == [WARRANTY]
    # with filtering off, "panel" matches the solar chunk too
    unfiltered = keyword_search(kb.engine, "panels guarantee", max_word_ratio=1.0)
    assert {h.chunk_id for h in unfiltered} == {SOLAR, WARRANTY}


def test_a_query_of_only_common_words_is_kept(kb):
    hits = keyword_search(kb.engine, "panels")  # every word is common, so the query stays as it was
    assert {h.chunk_id for h in hits} == {SOLAR, WARRANTY}


def test_vector_ranking_and_similarity(kb):
    query = mix((SOLAR, 0.8), (WARRANTY, 0.6))  # unit length
    hits = vector_search(kb.engine, query)
    assert [h.chunk_id for h in hits[:2]] == [SOLAR, WARRANTY]
    assert hits[0].similarity == pytest.approx(0.8, abs=1e-4)
    assert hits[1].similarity == pytest.approx(0.6, abs=1e-4)
    assert [h.vector_rank for h in hits[:2]] == [1, 2]


def test_vector_search_excludes_policy(kb):
    hits = vector_search(kb.engine, mix((POLICY, 1.0)))  # the exact vector of the policy chunk
    assert POLICY not in [h.chunk_id for h in hits]
    assert len(hits) == len(CHUNKS) - 1


def test_hybrid_fuses_both_searches(kb):
    kb.embedder.query_vector = mix((SOLAR, 0.8), (WARRANTY, 0.6))
    result = search("guarantee", k=2, mode="hybrid", engine=kb.engine, embedder=kb.embedder, min_similarity=0.5)
    assert (result.mode, result.degraded, result.no_context) == ("hybrid", False, False)
    # warranty: keyword rank 1 + vector rank 2 beats solar: vector rank 1 only
    assert [c.chunk_id for c in result.chunks] == [WARRANTY, SOLAR]
    assert result.chunks[0].keyword_rank == 1 and result.chunks[0].vector_rank == 2


def test_no_context_when_nothing_matches(kb):
    kb.embedder.query_vector = mix((POLICY, 1.0))  # only the policy chunk is close, and it is excluded
    result = search("weather in Paris", mode="hybrid", engine=kb.engine, embedder=kb.embedder, min_similarity=0.5)
    assert result.no_context is True


def test_search_falls_back_to_keyword_when_embedding_fails(kb):
    class Broken:
        def embed_query(self, text):
            raise EmbeddingError("down")

    result = search("guarantee", mode="hybrid", engine=kb.engine, embedder=Broken())
    assert (result.mode, result.degraded) == ("keyword", True)
    assert [c.chunk_id for c in result.chunks] == [WARRANTY]


def test_service_areas_load_from_the_table(kb):
    areas = ServiceAreas.from_db(kb.engine)
    assert sorted(areas.cities) == ["Delhi", "Lucknow", "Noida"]
    assert areas.lookup("noida").regions == ["Uttar Pradesh", "Delhi NCR"]
    assert areas.lookup("Lucknw").matched_city == "Lucknow"
    assert areas.lookup("Pune").status == "unverified"
    assert areas.rows() == ServiceAreas.from_sections(SECTIONS).rows()


def test_health_reports_chunk_count(kb, client, use_engine):
    use_engine(kb.engine)
    body = client.get("/api/health").json()
    assert body["kb_chunks"] == len(CHUNKS)


def test_health_kb_chunks_is_null_when_table_is_missing(client, use_engine, test_engine):
    with test_engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS kb_chunks"))
    use_engine(test_engine)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["kb_chunks"] is None


def test_kb_hints_include_the_package_guidance_rows(test_engine):
    # T13a: the KB Section 9 rows, read once with the titles and summaries (fake one-hot vectors, test database)
    from app.agent import understand

    guidance = Section(
        9, "Reference Package Guidance", 3,
        tables=[Table(3, ["Customer Signal", "Potential Reference Offering", "Do Not Assume"],
                      [["Moderate residential requirement", "Plan A", "Do not assume 3 kW is suitable."],
                       ["Higher residential requirement", "Plan B", "Do not assume 5 kW is sufficient."]])],
    )

    class OneHot:
        model, dim = "fake-embed", DIM

        def embed_documents(self, texts):
            return [[1.0] + [0.0] * (DIM - 1) for _ in texts]

    ingest_sections(SECTIONS + [guidance], "abc123def456", test_engine, OneHot())
    understand._hints_cache.pop(id(test_engine), None)
    try:
        rows = understand.load_kb_hints(test_engine).guidance
    finally:
        understand._hints_cache.pop(id(test_engine), None)
    assert [r["section_no"] for r in rows] == [9, 9]
    assert {r["chunk_type"] for r in rows} == {"table_row"}
    assert any("Plan A" in r["content"] and "3 kW" in r["content"] for r in rows)

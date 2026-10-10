"""Retriever logic that needs no database or network: fusion, gate and the keyword fallback."""
import pytest

from app.rag import retriever
from app.rag.retriever import RetrievedChunk, drop_common_words, has_context, or_query, rrf_fuse, search


def hit(chunk_id, **fields):
    return RetrievedChunk(chunk_id, 1, "Title", 1, "prose", f"content of {chunk_id}", **fields)


def test_rrf_scores_and_order():
    keyword = [hit("a", keyword_score=2.0), hit("b", keyword_score=1.0)]
    vector = [hit("b", similarity=0.9), hit("c", similarity=0.8)]
    fused = rrf_fuse(keyword, vector, keyword_weight=1.0)
    scores = {c.chunk_id: c.rrf_score for c in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)  # keyword rank 2 + vector rank 1
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert [c.chunk_id for c in fused] == ["b", "a", "c"]


def test_rrf_keyword_weight_scales_only_the_keyword_list():
    keyword = [hit("a", keyword_score=2.0), hit("b", keyword_score=1.0)]
    vector = [hit("b", similarity=0.9), hit("c", similarity=0.8)]
    scores = {c.chunk_id: c.rrf_score for c in rrf_fuse(keyword, vector, keyword_weight=0.5)}
    assert scores["a"] == pytest.approx(0.5 / 61)  # keyword rank 1, half weight
    assert scores["b"] == pytest.approx(0.5 / 62 + 1 / 61)  # keyword rank 2 (half) + vector rank 1
    assert scores["c"] == pytest.approx(1 / 62)  # vector only: unchanged


def test_rrf_default_weight_lets_vector_win_a_rank_tie():
    # rank 1 in the keyword list vs rank 1 in the vector list: the vector hit comes first
    fused = rrf_fuse([hit("a", keyword_score=2.0)], [hit("z", similarity=0.9)])
    assert [c.chunk_id for c in fused] == ["z", "a"]


def test_rrf_keeps_ranks_and_scores_from_both_lists():
    fused = rrf_fuse([hit("a", keyword_score=2.0)], [hit("a", similarity=0.7)])
    only = fused[0]
    assert (only.keyword_rank, only.keyword_score) == (1, 2.0)
    assert (only.vector_rank, only.similarity) == (1, 0.7)


def test_rrf_does_not_change_its_inputs():
    keyword = [hit("a", keyword_score=2.0)]
    vector = [hit("a", similarity=0.7)]
    rrf_fuse(keyword, vector)
    assert keyword[0].rrf_score is None and keyword[0].vector_rank is None
    assert vector[0].rrf_score is None and vector[0].keyword_rank is None


def test_rrf_ties_break_by_chunk_id():
    fused = rrf_fuse([hit("z")], [hit("m")], keyword_weight=1.0)  # both rank 1, same weight
    assert [c.chunk_id for c in fused] == ["m", "z"]


def test_rrf_with_one_empty_list():
    assert [c.chunk_id for c in rrf_fuse([], [hit("a"), hit("b")])] == ["a", "b"]
    assert rrf_fuse([], []) == []


# ---- common-word filter (pure functions) ----

# 4 chunks in total; "panel" is in 2 of them (50%), "guarante" in 1 (25%), "solar" in 1.
COUNTS = {"panel": 2, "guarante": 1, "solar": 1}


def test_drop_common_words_removes_words_above_the_ratio():
    assert drop_common_words(["panel", "guarante"], COUNTS, 4, 0.30) == ["guarante"]


def test_drop_common_words_keeps_words_at_the_ratio():
    # "panel" is in exactly 50% of the chunks, which is not more than 50%
    assert drop_common_words(["panel", "guarante"], COUNTS, 4, 0.50) == ["panel", "guarante"]


def test_drop_common_words_keeps_the_original_when_every_word_is_common():
    assert drop_common_words(["panel"], COUNTS, 4, 0.30) == ["panel"]


def test_drop_common_words_keeps_unknown_words():
    # "zebra" is in no chunk: it cannot add noise, and it may be the word that matters
    assert drop_common_words(["panel", "zebra"], COUNTS, 4, 0.30) == ["zebra"]


def test_drop_common_words_without_stats_or_words():
    assert drop_common_words(["panel", "guarante"], {}, 0, 0.30) == ["panel", "guarante"]  # total unknown
    assert drop_common_words([], COUNTS, 4, 0.30) == []


def test_drop_common_words_does_not_change_its_input():
    words = ["panel", "guarante"]
    drop_common_words(words, COUNTS, 4, 0.30)
    assert words == ["panel", "guarante"]


def test_or_query():
    assert or_query(["provid", "delhi"]) == "'provid' | 'delhi'"
    assert or_query(["delhi"]) == "'delhi'"
    assert or_query([]) == ""


def test_or_query_escapes_quotes_and_backslashes():
    assert or_query(["o'brien"]) == "'o''brien'"
    assert or_query(["a\\b"]) == "'a\\\\b'"


def test_gate():
    assert has_context(None, 1, 0.5)  # keyword-only fallback: a keyword hit is enough
    assert not has_context(None, 0, 0.5)
    assert has_context(0.6, 0, 0.5)  # similar enough
    assert has_context(0.5, 0, 0.5)  # exactly at the threshold
    assert not has_context(0.4, 0, 0.5)  # weak vector and no keyword hit


def test_gate_ignores_keyword_hits_when_a_vector_search_ran():
    # "Who won the India vs Australia cricket match?" matches the word "India" in the KB, but is off-topic
    assert not has_context(0.4, 3, 0.5)


class BrokenEmbedder:
    def embed_query(self, text):
        raise TimeoutError("slow")


class GoodEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2]


@pytest.fixture
def fake_searches(monkeypatch):
    """Replace the two SQL searches so search() runs without a database."""
    monkeypatch.setattr(
        retriever, "keyword_search",
        lambda engine, q, limit=10: [hit("k1", keyword_rank=1, keyword_score=0.5)],
    )
    monkeypatch.setattr(
        retriever, "vector_search",
        lambda engine, v, limit=10: [hit("v1", vector_rank=1, similarity=0.9)],
    )


@pytest.mark.parametrize("mode", ["hybrid", "vector"])
def test_embedding_failure_falls_back_to_keyword(fake_searches, mode):
    result = search("Do you install in Delhi?", mode=mode, engine=object(), embedder=BrokenEmbedder())
    assert result.mode == "keyword"
    assert result.degraded is True
    assert [c.chunk_id for c in result.chunks] == ["k1"]
    assert result.no_context is False  # the keyword hit still counts as context


def test_hybrid_when_embedding_works(fake_searches):
    result = search("q", mode="hybrid", engine=object(), embedder=GoodEmbedder())
    assert (result.mode, result.degraded) == ("hybrid", False)
    assert {c.chunk_id for c in result.chunks} == {"k1", "v1"}


def test_keyword_mode_never_embeds(fake_searches):
    result = search("q", mode="keyword", engine=object(), embedder=BrokenEmbedder())
    assert (result.mode, result.degraded) == ("keyword", False)


def test_hybrid_ignores_keyword_hits_when_similarity_is_weak(monkeypatch):
    monkeypatch.setattr(
        retriever, "keyword_search", lambda engine, q, limit=10: [hit("k1", keyword_score=0.5)]
    )
    monkeypatch.setattr(
        retriever, "vector_search", lambda engine, v, limit=10: [hit("v1", similarity=0.3)]
    )
    result = search("q", mode="hybrid", engine=object(), embedder=GoodEmbedder(), min_similarity=0.5)
    assert result.no_context is True
    assert {c.chunk_id for c in result.chunks} == {"k1", "v1"}  # near misses are still returned


def test_vector_mode_gate(monkeypatch):
    monkeypatch.setattr(
        retriever, "vector_search",
        lambda engine, v, limit=10: [hit("v1", vector_rank=1, similarity=0.3)],
    )
    result = search("q", mode="vector", engine=object(), embedder=GoodEmbedder(), min_similarity=0.5)
    assert result.no_context is True
    assert [c.chunk_id for c in result.chunks] == ["v1"]  # near misses are still returned


def test_k_limits_results(monkeypatch):
    monkeypatch.setattr(
        retriever, "keyword_search",
        lambda engine, q, limit=10: [hit(f"k{i}", keyword_rank=i) for i in range(1, 8)],
    )
    assert len(search("q", k=3, mode="keyword", engine=object()).chunks) == 3


def test_blank_query_and_bad_mode():
    result = search("   ", engine=object())
    assert result.no_context is True and result.chunks == []
    with pytest.raises(ValueError):
        search("q", mode="fuzzy", engine=object())

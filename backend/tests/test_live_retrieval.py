"""Live retrieval check: real Gemini embeddings against the ingested database. Reads only.

Run: python -m pytest -m live
"""
import pytest

from app.core.config import get_settings
from app.core.db import get_engine
from app.rag.retriever import count_kb_chunks, search

pytestmark = pytest.mark.live


def test_guarantee_question_finds_the_warranty_section():
    settings = get_settings()
    if not (settings.database_url and settings.google_api_key):
        pytest.skip("DATABASE_URL and GOOGLE_API_KEY are needed")
    if not count_kb_chunks(get_engine()):
        pytest.skip("kb_chunks is empty; run python -m app.rag.ingest first")
    result = search("Is there a guarantee on the panels?", k=3)
    assert not result.degraded, "embedding failed, so this was keyword-only"
    assert 11 in [c.section_no for c in result.chunks]  # KB Section 11

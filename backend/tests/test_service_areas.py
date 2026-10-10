"""Service-area lookup. PDF-based tests are skipped when the KB PDF is missing."""
import pytest

from app.core.config import get_settings
from app.services.service_areas import ServiceAreas

needs_pdf = pytest.mark.skipif(not get_settings().kb_file.exists(), reason="KB PDF not found")

SMALL_ROWS = [
    {"Region": "Uttar Pradesh", "Reference Locations": "Lucknow, Noida"},
    {"Region": "Delhi NCR", "Reference Locations": "Delhi, Noida"},
]


@pytest.fixture(scope="module")
def kb_areas():
    return ServiceAreas.from_kb(get_settings().kb_file)


@needs_pdf
def test_17_unique_cities(kb_areas):
    assert len(kb_areas.cities) == 17


@needs_pdf
def test_lucknow_listed_exact(kb_areas):
    r = kb_areas.lookup("Lucknow")
    assert (r.status, r.matched_city, r.match) == ("listed", "Lucknow", "exact")
    assert r.regions == ["Uttar Pradesh"]
    assert r.source == "KB Section 10"


@needs_pdf
def test_gurgaon_listed_via_alias(kb_areas):
    r = kb_areas.lookup("Gurgaon")
    assert (r.status, r.matched_city, r.match) == ("listed", "Gurugram", "alias")
    assert r.regions == ["Delhi NCR"]


@needs_pdf
def test_allahabad_and_new_delhi_aliases(kb_areas):
    assert kb_areas.lookup("Allahabad").matched_city == "Prayagraj"
    assert kb_areas.lookup("New Delhi").matched_city == "Delhi"


@needs_pdf
def test_noida_listed_in_two_regions(kb_areas):
    r = kb_areas.lookup("noida")
    assert r.status == "listed"
    assert r.regions == ["Uttar Pradesh", "Delhi NCR"]


@needs_pdf
def test_typo_listed_via_fuzzy(kb_areas):
    r = kb_areas.lookup("Lucknw")
    assert (r.status, r.matched_city, r.match) == ("listed", "Lucknow", "fuzzy")


@needs_pdf
def test_pune_unverified(kb_areas):
    r = kb_areas.lookup("Pune")
    assert (r.status, r.matched_city, r.match, r.regions) == ("unverified", None, "none", [])
    assert r.source == "KB Section 10"


@needs_pdf
def test_unrelated_words_not_fuzzy_matched(kb_areas):
    for name in ("Mumbai", "Kanpur Dehat Town", "Delhi Cantt Area", "Kota Bharu"):
        assert kb_areas.lookup(name).status == "unverified"


# The tests below use a small made-up table, so they also run without the PDF.
def test_state_narrows_regions_only():
    areas = ServiceAreas.from_rows(SMALL_ROWS)
    assert areas.lookup("Noida", "Uttar Pradesh").regions == ["Uttar Pradesh"]
    assert areas.lookup("Noida", "Delhi NCR").regions == ["Delhi NCR"]
    # An unrelated state does not change status or hide the regions.
    r = areas.lookup("Noida", "Kerala")
    assert r.status == "listed" and len(r.regions) == 2


def test_alias_target_must_be_in_the_list():
    areas = ServiceAreas.from_rows(SMALL_ROWS)  # Gurugram is not listed here
    assert areas.lookup("Gurgaon").status == "unverified"
    assert areas.lookup("New Delhi").matched_city == "Delhi"


def test_empty_and_punctuation_input():
    areas = ServiceAreas.from_rows(SMALL_ROWS)
    assert areas.lookup("").status == "unverified"
    assert areas.lookup("  ").status == "unverified"
    assert areas.lookup("  LUCKNOW! ").match == "exact"

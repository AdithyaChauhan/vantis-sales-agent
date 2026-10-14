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
    for name in ("Mumbai", "Kanpur Dehat Town", "Navi Mumbai", "Greater Noida West"):
        assert kb_areas.lookup(name).status == "unverified"


@needs_pdf
@pytest.mark.parametrize("place, city", [
    ("Sector 62, Noida", "Noida"), ("Noida sector 18", "Noida"), ("Gomti Nagar, Lucknow", "Lucknow"),
    ("Indirapuram, Ghaziabad", "Ghaziabad"), ("South Delhi", "Delhi"), ("DLF Phase 3, Gurugram", "Gurugram"),
    ("Delhi Cantt Area", "Delhi"), ("DLF Phase 3, Gurgaon", "Gurugram"),
])
def test_a_locality_of_a_listed_city_is_that_city(kb_areas, place, city):
    """T13d (Adithya): a locality with a listed city's name counts as that city."""
    r = kb_areas.lookup(place)
    assert (r.status, r.matched_city, r.match) == ("listed", city, "locality")


@needs_pdf
@pytest.mark.parametrize("place", [
    "Greater Noida", "Noida Extension", "Kanpur Dehat", "near Lucknow", "Lucknow ke paas ek gaon", "30 km from Lucknow",
    "outskirts of Jaipur", "Lucknow district", "close to Kanpur", "Lucknow ke aas paas",
])
def test_a_separate_place_or_a_landmark_phrase_is_unverified(kb_areas, place):
    """T13d (Adithya): only a known separate place that contains a listed name, or a landmark phrase around it, makes it
    unverified; never fuzzy-matched to the listed city."""
    r = kb_areas.lookup(place)
    assert (r.status, r.matched_city, r.match) == ("unverified", None, "lookalike")


@needs_pdf
def test_aliases_and_typos_still_come_before_the_fuzzy_match(kb_areas):
    assert kb_areas.lookup("New Delhi").match == "alias"
    assert (kb_areas.lookup("Dehli").matched_city, kb_areas.lookup("Lucknw").matched_city) == ("Delhi", "Lucknow")


@needs_pdf
@pytest.mark.parametrize("text, places", [
    ("aap log noida me kaam karte ho", ["Noida"]),
    ("hi, mera ghar Greater Noida me hai", ["Greater Noida"]),
    ("Lucknow ke paas ek gaon hai mera", ["Lucknow ke paas"]),
    ("30 km from Lucknow, also an office in Kanpur", ["30 km from Lucknow", "Kanpur"]),
    ("I live in Dehli", ["Delhi"]),
    ("Sector 62, Noida, and a Gurgaon office", ["Noida", "Gurugram"]),
    ("Karna", []),  # never fuzzy: "Karna" is not Karnal
    ("I live in Pune", []),
])
def test_find_in_text(kb_areas, text, places):
    assert kb_areas.find_in_text(text) == places


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


# ---------- T12b: one swapped pair of letters ----------

from app.services.service_areas import one_swap_apart  # noqa: E402

TYPO_AREAS = ServiceAreas({"Delhi": ["Delhi NCR"], "Lucknow": ["Uttar Pradesh"], "Jaipur": ["Rajasthan"],
                           "Gurugram": ["Delhi NCR"], "Kota": ["Rajasthan"]})


@pytest.mark.parametrize("typed, city", [("Dehli", "Delhi"), ("dlehi", "Delhi"), ("Jiapur", "Jaipur")])
def test_one_swapped_pair_of_letters_matches(typed, city):
    r = TYPO_AREAS.lookup(typed)
    assert (r.status, r.matched_city, r.match) == ("listed", city, "typo")


@pytest.mark.parametrize("typed", ["Raipur", "Pune", "Kanpur", "Kotb", "Ktoa x"])
def test_one_other_change_is_not_a_typo_match(typed):
    # Raipur is one letter from Jaipur, and a real city that is not on the list
    assert TYPO_AREAS.lookup(typed).status == "unverified"


def test_aliases_and_fuzzy_matches_come_first():
    assert TYPO_AREAS.lookup("Gurgaon").match == "alias"
    assert TYPO_AREAS.lookup("Lucknwo").match == "fuzzy"


def test_one_swap_apart_rules():
    assert one_swap_apart("dehli", "delhi")
    assert not one_swap_apart("delhi", "delhi")
    assert not one_swap_apart("kota", "ktoa x")
    assert not one_swap_apart("ab", "ba")  # too short to trust
    assert not one_swap_apart("raipur", "jaipur")


@needs_pdf
def test_dehli_is_delhi_with_the_real_kb(kb_areas):
    r = kb_areas.lookup("Dehli")
    assert (r.status, r.matched_city) == ("listed", "Delhi")
    assert kb_areas.lookup("Raipur").status == "unverified"


@needs_pdf
@pytest.mark.parametrize("place", ["Lucknow se 30 km door", "Lucknow se thoda door", "Kanpur se bahar"])
def test_the_hinglish_distance_form_is_a_landmark(kb_areas, place):
    assert kb_areas.lookup(place).status == "unverified"


@needs_pdf
def test_the_hinglish_distance_form_in_a_message(kb_areas):
    assert kb_areas.find_in_text("mera gaon Lucknow se 30 km door hai") == ["Lucknow se 30 km door"]
    assert kb_areas.find_in_text("main Lucknow se hu") == ["Lucknow"]  # "from Lucknow": the city itself

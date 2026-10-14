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
@pytest.mark.parametrize("place", [
    "Sector 62, Noida", "Noida sector 18", "Gomti Nagar, Lucknow", "South Delhi", "DLF Phase 3, Gurgaon",
])
def test_a_locality_left_in_the_name_is_never_guessed(kb_areas, place):
    """T13e: lookup() checks the city alone; the understand step splits a locality off (tests/test_agent_turn_rules.py).
    A locality left in the name is unverified, the safe side, never guessed as the city."""
    assert kb_areas.lookup(place).status == "unverified"


@needs_pdf
@pytest.mark.parametrize("place", [
    "Greater Noida", "Noida Extension", "Kanpur Dehat", "near Lucknow", "Lucknow ke paas ek gaon", "30 km from Lucknow",
    "outskirts of Jaipur", "Lucknow district", "close to Kanpur", "Lucknow ke aas paas",
])
def test_a_separate_place_or_a_landmark_phrase_is_unverified(kb_areas, place):
    """T13d (Adithya): a separate place that contains a listed name, or a landmark phrase around it, is never the listed
    city. T13e: no phrase list in lookup() any more; plain whole-string fuzz.ratio never scores them close."""
    r = kb_areas.lookup(place)
    assert (r.status, r.matched_city) == ("unverified", None)


@needs_pdf
def test_every_listed_city_matches_a_typo_but_never_a_longer_name(kb_areas):
    """T13e (Adithya): a dropped, a doubled or a swapped pair of letters still matches the city; the city with a word in
    front or behind stays unverified (only the alias "New Delhi" is Delhi). It pins the plain fuzz.ratio scorer: WRatio,
    extractOne's default, scores "Greater Noida" close to Noida."""
    for city in kb_areas.cities:
        name = city.lower()
        typos = [name[:i] + name[i + 1:] for i in range(len(name))] + [name[:i] + name[i] + name[i:]
                                                                        for i in range(len(name))]
        typos += [name[:i] + name[i + 1] + name[i] + name[i + 2:] for i in range(len(name) - 1)
                  if name[i] != name[i + 1]]
        for typo in typos:
            assert kb_areas.lookup(typo).matched_city == city, (city, typo)
        for longer in (f"Greater {city}", f"New {city}", f"Old {city}", f"{city} Extension", f"{city} Dehat",
                       f"{city} Cantt", f"{city} Rural", f"near {city}", f"the area near {city}"):
            if longer != "New Delhi":
                assert kb_areas.lookup(longer).status == "unverified", longer
    assert kb_areas.lookup("New Delhi").matched_city == "Delhi"


REGIONS = {"Delhi NCR": "Delhi NCR", "NCR": "Delhi NCR", "Delhi-NCR": "Delhi NCR", "Uttar Pradesh": "Uttar Pradesh",
           "UP": "Uttar Pradesh", "U.P.": "Uttar Pradesh", "UP state": "Uttar Pradesh", "Punjab": "Punjab",
           "Haryana": "Haryana", "Rajasthan": "Rajasthan"}


@needs_pdf
@pytest.mark.parametrize("name, region", REGIONS.items())
def test_a_region_is_never_a_listed_city(kb_areas, name, region):
    """T13e: "I'm in Delhi NCR" got "Vantis currently covers Delhi NCR". A KB Section 10 region (or its short form) is a
    region: no matched city, with the region's listed cities from the KB."""
    r = kb_areas.lookup(name)
    assert (r.status, r.matched_city, r.regions) == ("region", None, [region])
    expected = {c for c in kb_areas.cities if region in kb_areas.city_regions[c]}
    assert r.region_cities and set(r.region_cities) == expected


@needs_pdf
@pytest.mark.parametrize("place, status, city", [
    ("Delhi", "listed", "Delhi"), ("New Delhi", "listed", "Delhi"), ("Noida, UP", "listed", "Noida"),
    ("Gurgaon, Haryana", "listed", "Gurugram"), ("Lucknow Uttar Pradesh", "listed", "Lucknow"),
    ("Mohali, Punjab", "unverified", None), ("Hariyana", "region", None),
])
def test_a_city_with_its_region_is_the_city(kb_areas, place, status, city):
    r = kb_areas.lookup(place)
    assert (r.status, r.matched_city) == (status, city)


@needs_pdf
@pytest.mark.parametrize("text, places", [
    ("I'm in Delhi NCR. Do you cover my area?", ["Delhi NCR"]),  # English
    ("Delhi NCR mein service hai kya?", ["Delhi NCR"]),  # Hinglish
    ("Do you install in NCR?", ["Delhi NCR"]),
    ("NCR me lagate ho?", ["Delhi NCR"]),
    ("I live in Uttar Pradesh", ["Uttar Pradesh"]),
    ("main Uttar Pradesh se hu", ["Uttar Pradesh"]),
    ("Do you work in UP?", ["Uttar Pradesh"]),
    ("UP mein solar lagwate ho kya?", ["Uttar Pradesh"]),
    ("Do you cover Punjab?", ["Punjab"]),
    ("Punjab mein installation karte ho?", ["Punjab"]),
    ("I'm from Haryana, can I get solar?", ["Haryana"]),
    ("Haryana me solar lagwana hai", ["Haryana"]),
    ("Is Rajasthan covered?", ["Rajasthan"]),
    ("Rajasthan se hu, aap aate ho?", ["Rajasthan"]),
    ("solar set up ka kharcha kitna hai", []),  # "up" in small letters is never Uttar Pradesh
    ("I live in Delhi", ["Delhi"]),
])
def test_find_in_text_reads_regions_in_the_backup(kb_areas, text, places):
    """Backup only (the understand step failed): a region in the raw message is the region, never the listed city
    inside its name."""
    assert kb_areas.find_in_text(text) == places


def test_regions_come_from_the_rows_too():
    areas = ServiceAreas.from_rows(SMALL_ROWS)
    r = areas.lookup("NCR")
    assert (r.status, set(r.region_cities)) == ("region", {"Delhi", "Noida"})
    assert areas.lookup("Uttar Pradesh").status == "region" and areas.lookup("Lucknow").status == "listed"


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
    ("Sector 62, Noida, and a Gurgaon office", ["Noida", "Gurugram"]),  # backup: a listed name in the words
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

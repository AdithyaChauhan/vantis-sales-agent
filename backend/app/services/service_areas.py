"""Service-area lookup. The cities and their regions come from KB Section 10, not from code.

T13e: lookup() checks one place name the understand step gave, the city alone by its official name (the understand step
puts anything finer, a sector or a village, in its own field): exact, alias, a region, fuzzy (whole string) or one
swapped pair of letters; anything else is unverified. Whether the customer's place is inside the city, only near it, a
whole region or unclear is the understand step's call (place_relation); the graph confirms only a place inside a listed
city. The phrase lists below (SEPARATE_PLACES and the landmark phrases) are only the backup for a turn whose understand
step failed: find_in_text reads the raw message with them.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.core.db import get_engine
from app.rag.models import ServiceArea
from app.rag.pdf_parser import Section, parse_kb, table_dicts

SOURCE = "KB Section 10"
FUZZY_MIN_SCORE = 85
# Renamed cities (general knowledge, not a business fact). Target must be in the KB list.
ALIASES = {"gurgaon": "Gurugram", "allahabad": "Prayagraj", "new delhi": "Delhi"}
# T13e: short forms of KB Section 10's regions (general knowledge, like ALIASES); a target must be a KB region
REGION_ALIASES = {"ncr": "Delhi NCR", "ncr delhi": "Delhi NCR", "up": "Uttar Pradesh", "u p": "Uttar Pradesh"}
# Words that may come with a region's name and add nothing ("the Delhi NCR region", "UP state")
REGION_FILLERS = {"region", "state", "area", "side", "belt", "zone", "the", "in", "of", "mein", "me", "mai", "se"}
# "UP" in a raw message only in capitals ("UP mein", "U.P."), never "set up" (backup only, find_in_text)
_UP_RE = re.compile(r"(?<![A-Za-z])U\.?\s?P(?![A-Za-z])")
# Backup only (the understand step failed, find_in_text), T13d: separate places whose names contain a listed city's
# name (general knowledge, like ALIASES); they are never that city: Greater Noida is not Noida
SEPARATE_PLACES = ("greater noida", "noida extension", "kanpur dehat")
# Backup only, T13d: a listed name used only as a landmark is another, unverified place ("near Lucknow", "30 km from
# Kanpur", "Lucknow district")
_LANDMARK_BEFORE = (r"(?:near(?:by)?|close\s+to|outside(?:\s+of)?|(?:on\s+the\s+)?outskirts\s+of|"
                    r"\d+(?:\.\d+)?\s*(?:km|kms|kilomet(?:er|re)s?)\s+(?:away\s+)?(?:from|outside(?:\s+of)?))")
_LANDMARK_AFTER = (r"(?:ke\s+(?:paas|pass|nazdeek|najdeek|aas\s*-?\s*paas|kareeb|qareeb|bahar)|district|"
                   r"se\s+(?:\d+(?:\.\d+)?\s*(?:km|kms|kilomet(?:er|re)s?)\s+)?(?:thoda\s+|kaafi\s+|bahut\s+)?"
                   r"(?:door|dur|bahar))")


@dataclass
class ServiceAreaResult:
    """Outcome of a city lookup."""

    query: str
    status: str  # "listed" | "unverified"
    matched_city: str | None
    regions: list[str] = field(default_factory=list)
    match: str = "none"  # "exact" | "alias" | "region" | "fuzzy" | "typo" | "none"
    source: str = SOURCE
    region_cities: list[str] = field(default_factory=list)  # T13e: a region's listed cities (status "region")


def one_swap_apart(a: str, b: str) -> bool:
    """True if b is a with two neighbouring letters swapped ("dehli" / "delhi"), for names of 4+ letters (T12b).

    fuzz.ratio scores such a swap only 80, below the cutoff. Only a swap counts, never any one-letter change:
    "Raipur" is one letter from "Jaipur", and Raipur is a real city that is not on the list.
    """
    if len(a) != len(b) or len(a) < 4 or a == b:
        return False
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse spaces."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def _name_re(name: str) -> str:
    """A place name as whole words, any spacing or case."""
    return r"\b" + r"\s+".join(re.escape(part) for part in name.split()) + r"\b"


def region_patterns(region: str) -> list[re.Pattern]:
    """Patterns for a KB region's name and its short forms (REGION_ALIASES) in a text, as whole words with any
    spacing, dash or slash; "UP" only in capitals, so "set up" never matches (T13e)."""
    keys = [_normalize(region)] + [alias for alias, target in REGION_ALIASES.items() if target == region]
    return [_UP_RE if key in ("up", "u p") else
            re.compile(r"\b" + r"[\s\-/&]+".join(re.escape(part) for part in key.split()) + r"\b", re.I)
            for key in keys]


def lookalike_place(text: str, name: str) -> str | None:
    """The phrase, when `name` (a listed city or alias) appears in `text` only inside a separate place ("Greater
    Noida") or a landmark phrase ("near Lucknow", "Lucknow ke paas"): that phrase is the customer's place, unverified.
    None when the name is used as the city itself at least once, or not at all (T13d)."""
    phrases, plain = [], False
    spans = []
    for place in SEPARATE_PLACES:
        spans += [(m.start(), m.end()) for m in re.finditer(_name_re(place), text, re.I)]
    landmark = re.compile(rf"{_LANDMARK_BEFORE}\s+{_name_re(name)}|{_name_re(name)}\s+{_LANDMARK_AFTER}", re.I)
    spans += [(m.start(), m.end()) for m in landmark.finditer(text)]
    for match in re.finditer(_name_re(name), text, re.I):
        inside = [(a, b) for a, b in spans if a <= match.start() and match.end() <= b]
        if inside:
            a, b = max(inside, key=lambda span: span[1] - span[0])
            phrases.append(" ".join(text[a:b].split()))
        else:
            plain = True
    return phrases[0] if phrases and not plain else None


class ServiceAreas:
    """Cities from KB Section 10 with their regions."""

    def __init__(self, city_regions: dict[str, list[str]]):
        self.city_regions = city_regions  # display name -> regions
        self._by_key = {_normalize(city): city for city in city_regions}
        self.regions: dict[str, list[str]] = {}  # T13e: region -> its cities, in the order the rows were written
        for city, regions in city_regions.items():
            for region in regions:
                self.regions.setdefault(region, []).append(city)

    @classmethod
    def from_rows(cls, rows: list[dict[str, str]]) -> "ServiceAreas":
        """Build from KB Section 10 table rows (keys: Region, Reference Locations)."""
        city_regions: dict[str, list[str]] = {}
        for row in rows:
            for city in row["Reference Locations"].split(","):
                city = city.strip()
                if city:
                    city_regions.setdefault(city, []).append(row["Region"])
        return cls(city_regions)

    @classmethod
    def from_sections(cls, sections: list[Section]) -> "ServiceAreas":
        """Build from the Section 10 table of already-parsed KB sections."""
        section = next((s for s in sections if s.number == 10), None)
        if section is None:
            raise ValueError("KB Section 10 (service areas) was not found in the parsed KB")
        return cls.from_rows([row for table in section.tables for row in table_dicts(table)])

    @classmethod
    def from_kb(cls, path: Path) -> "ServiceAreas":
        """Parse the KB PDF and build from its Section 10 table."""
        return cls.from_sections(parse_kb(path))

    @classmethod
    def from_db(cls, engine: Engine) -> "ServiceAreas":
        """Build from the service_areas table, in the order the rows were written."""
        with Session(engine) as session:
            found = session.execute(
                select(ServiceArea.city, ServiceArea.region).order_by(ServiceArea.id)
            ).all()
        city_regions: dict[str, list[str]] = {}
        for city, region in found:
            city_regions.setdefault(city, []).append(region)
        return cls(city_regions)

    def rows(self) -> list[tuple[str, str]]:
        """Every (city, region) pair, the shape stored in the service_areas table."""
        return [(city, region) for city, regions in self.city_regions.items() for region in regions]

    @property
    def cities(self) -> list[str]:
        """All listed cities, each once."""
        return list(self.city_regions)

    def _names(self) -> dict[str, str]:
        """Every name that means a listed city, normalized: the cities and the aliases -> the listed city."""
        names = dict(self._by_key)
        names.update({alias: self._by_key[_normalize(target)] for alias, target in ALIASES.items()
                      if _normalize(target) in self._by_key})
        return names

    def _region_names(self) -> dict[str, str]:
        """Every name that means a KB region, normalized: the regions and their short forms -> the region."""
        names = {_normalize(region): region for region in self.regions}
        names.update({alias: target for alias, target in REGION_ALIASES.items() if target in self.regions})
        return names

    def _without_region_names(self, key: str) -> tuple[str, list[str]]:
        """(the normalized name without KB region names and fillers, the regions it named)."""
        words = f" {key} "
        found = []
        for name, region in sorted(self._region_names().items(), key=lambda item: -len(item[0])):
            if f" {name} " in words:
                words = words.replace(f" {name} ", " ")
                if region not in found:
                    found.append(region)
        rest = [word for word in words.split() if word not in REGION_FILLERS] if found else words.split()
        return " ".join(rest), found

    def region_of(self, name: str) -> list[str]:
        """The KB regions a place name means when it names only regions (T13e): "Delhi NCR", "NCR", "UP state",
        "Punjab"; [] when it names a city or anything else ("Noida, UP" is a city)."""
        rest, found = self._without_region_names(_normalize(name))
        return found if found and not rest else []

    def _region_result(self, query: str, regions: list[str]) -> ServiceAreaResult:
        cities = list(dict.fromkeys(city for region in regions for city in self.regions[region]))
        return ServiceAreaResult(query, "region", None, list(regions), "region", region_cities=cities)

    def _inside(self, text: str) -> list[tuple[int, str, str]]:
        """(position, name as written, listed city) for every listed name or alias in `text` as whole words; a name
        inside a longer one ("delhi" in "new delhi") is left out."""
        found = []
        for name, city in self._names().items():
            found += [(m.start(), m.end(), m.group(0), city) for m in re.finditer(_name_re(name), text, re.I)]
        kept = [f for f in found if not any(o[0] <= f[0] and f[1] <= o[1] and (o[1] - o[0]) > (f[1] - f[0])
                                            for o in found)]
        return [(start, written, city) for start, _, written, city in sorted(kept)]

    def lookup(self, city: str, state: str | None = None) -> ServiceAreaResult:
        """Check one place name, the city alone (T13e): exact, alias, a KB region ("Delhi NCR", "UP": status
        "region" with its cities), fuzzy (plain whole-string ratio >= 85, so "Greater Noida" never scores close to
        Noida) and one swapped pair of letters; else unverified.

        `state` only narrows the returned regions; it never changes the status.
        """
        key = _normalize(city)
        if key in self._by_key:
            return self._result(city, self._by_key[key], "exact", state)
        alias_target = ALIASES.get(key)
        if alias_target and _normalize(alias_target) in self._by_key:
            return self._result(city, self._by_key[_normalize(alias_target)], "alias", state)
        regions = self.region_of(city)
        if regions:
            return self._region_result(city, regions)
        rest, named = self._without_region_names(key)
        if named and rest:  # a city with its region or state ("Noida, UP"): the city alone, the region as the hint
            found = self.lookup(rest, state or named[0])
            return ServiceAreaResult(city, found.status, found.matched_city, found.regions, found.match)
        if key:
            region_keys = {_normalize(region): region for region in self.regions}
            candidates = list(self._by_key) + list(ALIASES) + list(region_keys)
            best = process.extractOne(key, candidates, scorer=fuzz.ratio, score_cutoff=FUZZY_MIN_SCORE)
            if best:
                name = best[0]
                if name in region_keys and name not in self._by_key:
                    return self._region_result(city, [region_keys[name]])
                target = self._by_key.get(name) or self._by_key.get(_normalize(ALIASES[name]))
                if target:
                    return self._result(city, target, "fuzzy", state)
            swapped = next((name for name in self._by_key if one_swap_apart(key, name)), None)
            if swapped:
                return self._result(city, self._by_key[swapped], "typo", state)
        return ServiceAreaResult(query=city, status="unverified", matched_city=None)

    def _regions_in_text(self, text: str) -> list[tuple[int, int, str]]:
        """(start, end, region) for every KB region named in a text; "UP" only in capitals."""
        return [(m.start(), m.end(), region) for region in self.regions
                for pattern in region_patterns(region) for m in pattern.finditer(text)]

    def region_around(self, text: str, name: str) -> str | None:
        """The KB region, when every mention of `name` in the customer's message is inside a mention of that region's
        name: "Delhi" in "I'm in Delhi NCR" is Delhi NCR, never the city Delhi (T13e). It matches KB data only (the
        region names and their short forms), like the package-name check, so it can only stop a confirmation, never
        make one. None when the name is used on its own anywhere in the message, or not at all."""
        mentions = [(m.start(), m.end()) for m in re.finditer(_name_re(name), text, re.I)] if name.strip() else []
        if not mentions:
            return None
        spans = self._regions_in_text(text)
        for region in dict.fromkeys(region for _, _, region in spans):
            inside = [(a, b) for a, b, r in spans if r == region]
            if all(any(a <= start and end <= b for a, b in inside) for start, end in mentions):
                return region
        return None

    def find_in_text(self, text: str) -> list[str]:
        """Backup only (T13d, for when the understand step failed): the places a raw message names, in the order
        they appear. Each listed city once, by exact name, alias or one swapped pair of letters ("Dehli"), never fuzzy
        (so "Karna" never becomes Karnal); a listed name used only inside a separate place or a landmark phrase gives
        that phrase instead ("Greater Noida", "near Lucknow"), which lookup() then finds unverified. T13e: a KB region
        gives the region ("Delhi NCR", "UP" -> Uttar Pradesh), and a listed name inside it ("Delhi" in "Delhi NCR") is
        not a city."""
        regions = self._regions_in_text(text)
        found: list[tuple[int, str]] = [(start, region) for start, _, region in regions]
        for start, written, city in self._inside(text):
            if any(a <= start < b for a, b, _ in regions):
                continue
            found.append((start, lookalike_place(text, written) or city))
        for match in re.finditer(r"[a-z]{4,}", text.lower()):
            swapped = next((name for name in self._by_key if " " not in name and one_swap_apart(match.group(0), name)),
                           None)
            if swapped:
                found.append((match.start(), self._by_key[swapped]))
        places = []
        for _, place in sorted(found):
            if place.lower() not in (p.lower() for p in places):
                places.append(place)
        return places

    def _result(self, query: str, city: str, match: str, state: str | None) -> ServiceAreaResult:
        regions = self.city_regions[city]
        if state:
            narrowed = [r for r in regions if _normalize(r) == _normalize(state)]
            regions = narrowed or regions
        return ServiceAreaResult(query, "listed", city, list(regions), match)


@lru_cache
def get_service_areas() -> ServiceAreas:
    """Cached lookup built from the service_areas table, so the server never needs the PDF.

    Raises RuntimeError (not cached) if the database is not set up or ingest has not run.
    """
    engine = get_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is not set; cannot load service areas")
    areas = ServiceAreas.from_db(engine)
    if not areas.cities:
        raise RuntimeError("service_areas table is empty; run python -m app.rag.ingest")
    return areas

"""Service-area lookup. The city list comes from KB Section 10, not from code."""
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


@dataclass
class ServiceAreaResult:
    """Outcome of a city lookup."""

    query: str
    status: str  # "listed" | "unverified"
    matched_city: str | None
    regions: list[str] = field(default_factory=list)
    match: str = "none"  # "exact" | "alias" | "fuzzy" | "none"
    source: str = SOURCE


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse spaces."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


class ServiceAreas:
    """Cities from KB Section 10 with their regions."""

    def __init__(self, city_regions: dict[str, list[str]]):
        self.city_regions = city_regions  # display name -> regions
        self._by_key = {_normalize(city): city for city in city_regions}

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

    def lookup(self, city: str, state: str | None = None) -> ServiceAreaResult:
        """Check a city: exact, then alias, then fuzzy (score >= 85), else unverified.

        `state` only narrows the returned regions; it never changes the status.
        """
        key = _normalize(city)
        if key in self._by_key:
            return self._result(city, self._by_key[key], "exact", state)
        alias_target = ALIASES.get(key)
        if alias_target and _normalize(alias_target) in self._by_key:
            return self._result(city, self._by_key[_normalize(alias_target)], "alias", state)
        if key:
            candidates = list(self._by_key) + list(ALIASES)
            best = process.extractOne(key, candidates, scorer=fuzz.ratio, score_cutoff=FUZZY_MIN_SCORE)
            if best:
                name = best[0]
                target = self._by_key.get(name) or self._by_key.get(_normalize(ALIASES[name]))
                if target:
                    return self._result(city, target, "fuzzy", state)
        return ServiceAreaResult(query=city, status="unverified", matched_city=None)

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

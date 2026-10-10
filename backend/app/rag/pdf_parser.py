"""Read the Vantis KB PDF into numbered sections, each with its text lines and tables.

Known limit: a table that continues onto the next page is not merged (the KB has none).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

HEADER_RE = re.compile(r"^VANTIS ENERGY SOLUTIONS\s+Participant Knowledge Base")
FOOTER_RE = re.compile(r"^Pradrix AI Engineer\b.*\bPage \d+$")
HEADING_RE = re.compile(r"^(\d{1,2})\. (.+)$")
BULLET_FIX = "(cid:127)"  # pdfplumber's name for the bullet glyph
ARROW_RE = re.compile(r"(?<= )fi(?= )")  # the PDF's arrow glyph is extracted as the letters "fi"
PARAGRAPH_GAP = 5.5  # gap in points above which a line starts a new paragraph (wrapped lines: 2-5)


@dataclass
class Line:
    """One line of text outside any table."""

    text: str
    page: int
    new_paragraph: bool


@dataclass
class Table:
    """A table: header cells plus data rows, all cells on a single line."""

    page: int
    header: list[str]
    rows: list[list[str]]


@dataclass
class Section:
    """One numbered KB section. `page` is the page its heading is on."""

    number: int
    title: str
    page: int
    lines: list[Line] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)


def clean_cell(value: str | None) -> str:
    """Collapse line breaks and repeated spaces inside a table cell."""
    return " ".join((value or "").split())


def table_dicts(table: Table) -> list[dict[str, str]]:
    """Map each data row to its header names."""
    return [dict(zip(table.header, row)) for row in table.rows]


def _page_items(page: "pdfplumber.page.Page", page_no: int) -> list[tuple[float, object]]:
    """Return (top, Line | Table) items of one page, without header and footer lines."""
    items: list[tuple[float, object]] = []
    found = page.find_tables()
    for t in found:
        grid = [[clean_cell(c) for c in row] for row in t.extract()]
        items.append((t.bbox[1], Table(page_no, grid[0], grid[1:])))

    def outside_tables(obj: dict) -> bool:
        # Keep a character only if its middle is not inside any table box.
        if "x0" not in obj or "top" not in obj:
            return True
        mid_x, mid_y = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        return not any(
            b[0] <= mid_x <= b[2] and b[1] <= mid_y <= b[3] for b in (t.bbox for t in found)
        )

    previous_bottom = None
    for raw in page.filter(outside_tables).extract_text_lines():
        text = " ".join(raw["text"].replace(BULLET_FIX, "•").split())
        text = ARROW_RE.sub("→", text)
        if not text or HEADER_RE.match(text) or FOOTER_RE.match(text):
            continue
        gap = None if previous_bottom is None else raw["top"] - previous_bottom
        items.append((raw["top"], Line(text, page_no, gap is None or gap > PARAGRAPH_GAP)))
        previous_bottom = raw["bottom"]
    return sorted(items, key=lambda item: item[0])


def parse_kb(path: Path) -> list[Section]:
    """Parse the KB PDF into sections. The cover page (page 1) is skipped."""
    sections: list[Section] = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages[1:], start=2):
            for _, item in _page_items(page, index):
                if isinstance(item, Line):
                    match = HEADING_RE.match(item.text)
                    # Only the next expected number counts as a heading.
                    if match and int(match.group(1)) == len(sections) + 1:
                        sections.append(Section(int(match.group(1)), match.group(2), index))
                        continue
                if not sections:
                    continue  # text before section 1 (none expected)
                if isinstance(item, Line):
                    sections[-1].lines.append(item)
                else:
                    sections[-1].tables.append(item)
    return sections

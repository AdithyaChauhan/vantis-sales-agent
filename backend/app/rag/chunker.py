"""Structure-aware chunking of parsed KB sections."""
import re
from dataclasses import dataclass, field

from app.rag.pdf_parser import Line, Section, table_dicts

TABLE_SECTIONS = {8, 9, 10, 11, 12}
# Tables that also get one chunk listing every row, so broad questions ("What systems do you
# offer?", "Where do you install?") match one chunk with the whole list (T5, docs/EVAL.md).
CATALOG_SECTION = 8  # Product & Service Catalog
SERVICE_AREA_SECTION = 10  # Reference Service Locations
PACKAGE_GUIDANCE_SECTION = 9  # Reference Package Guidance (T13a: in the context on a turn that gives the monthly bill)
SUMMARY_SECTIONS = {CATALOG_SECTION, SERVICE_AREA_SECTION}
SUMMARY_LABEL = "Summary of all rows"
FAQ_SECTION = 15
POLICY_SECTIONS = {16, 17, 18, 19}
MAX_PROSE_CHARS = 1200
MAX_SLUG_CHARS = 50


@dataclass
class Chunk:
    """One retrievable piece of the KB."""

    chunk_id: str
    section_no: int
    section_title: str
    page: int
    chunk_type: str  # prose | table_row | table_summary | faq | policy
    content: str
    metadata: dict = field(default_factory=dict)

    def embed_title(self) -> str:
        """Title for the embedding text: the section, plus the row or question label. No '|'."""
        title = f"KB Section {self.section_no}: {self.section_title}"
        label = self.metadata.get("label")
        if label:
            title += f" - {label}"
        return title.replace("|", "/")

    def embed_text(self) -> str:
        """Text to embed, in the document format gemini-embedding-2 expects."""
        return f"title: {self.embed_title()} | text: {self.content}"


def slugify(text: str, max_chars: int = MAX_SLUG_CHARS) -> str:
    """Lowercase letters and digits joined by '-', cut at a word boundary."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_chars:
        slug = slug[:max_chars].rsplit("-", 1)[0]
    return slug


def _paragraphs(lines: list[Line]) -> list[str]:
    """Join wrapped lines into paragraphs; every bullet line stays on its own line."""
    paragraphs: list[str] = []
    for line in lines:
        is_bullet = line.text.startswith("•")
        if paragraphs and not line.new_paragraph and not is_bullet:
            paragraphs[-1] += " " + line.text
        else:
            paragraphs.append(line.text)
    return paragraphs


def _split_text(paragraphs: list[str], limit: int = MAX_PROSE_CHARS) -> list[str]:
    """Group paragraphs into pieces of at most `limit` characters, cutting between paragraphs."""
    pieces: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        if current and len("\n".join(current + [para])) > limit:
            pieces.append("\n".join(current))
            current = []
        current.append(para)
    if current:
        pieces.append("\n".join(current))
    return pieces


def _table_chunks(section: Section) -> list[Chunk]:
    chunks = []
    for table in section.tables:
        for row in table_dicts(table):
            label = next(iter(row.values()))
            chunks.append(
                Chunk(
                    chunk_id=f"s{section.number:02d}-{slugify(label)}",
                    section_no=section.number,
                    section_title=section.title,
                    page=table.page,
                    chunk_type="table_row",
                    content="\n".join(f"{key}: {value}" for key, value in row.items()),
                    metadata={"label": label, "row": row},
                )
            )
    return chunks


def _summary_chunks(section: Section) -> list[Chunk]:
    """One chunk per table listing every row: a line naming the columns, then one line per row."""
    chunks = []
    for table in section.tables:
        rows = table_dicts(table)
        if not rows:
            continue
        lines = [f"All rows ({'; '.join(table.header)}):"]
        lines += ["- " + "; ".join(row.values()) for row in rows]
        chunks.append(
            Chunk(
                chunk_id=f"s{section.number:02d}-table-summary",
                section_no=section.number,
                section_title=section.title,
                page=table.page,
                chunk_type="table_summary",
                content="\n".join(lines),
                metadata={"label": SUMMARY_LABEL, "rows": len(rows)},
            )
        )
    return chunks


def _faq_chunks(section: Section) -> list[Chunk]:
    """A line ending in '?' starts a question; the lines after it are the answer."""
    qa: list[tuple[str, int, list[Line]]] = []
    for line in section.lines:
        if line.text.endswith("?"):
            qa.append((line.text, line.page, []))
        elif qa:
            qa[-1][2].append(line)
    return [
        Chunk(
            chunk_id=f"s{section.number:02d}-{slugify(question)}",
            section_no=section.number,
            section_title=section.title,
            page=page,
            chunk_type="faq",
            content=f"Q: {question}\nA: {' '.join(a.text for a in answer)}",
            metadata={"label": question, "question": question},
        )
        for question, page, answer in qa
    ]


def _prose_chunks(section: Section) -> list[Chunk]:
    if not section.lines:
        return []
    chunk_type = "policy" if section.number in POLICY_SECTIONS else "prose"
    base = f"s{section.number:02d}-{slugify(section.title)}"
    pieces = _split_text(_paragraphs(section.lines))
    return [
        Chunk(
            chunk_id=base if i == 1 else f"{base}-{i}",
            section_no=section.number,
            section_title=section.title,
            page=section.page,
            chunk_type=chunk_type,
            content=piece,
        )
        for i, piece in enumerate(pieces, start=1)
    ]


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """Turn parsed sections into chunks. Raises if two chunks get the same id."""
    chunks: list[Chunk] = []
    for section in sections:
        if section.number in TABLE_SECTIONS:
            chunks += _table_chunks(section)
            if section.number in SUMMARY_SECTIONS:
                chunks += _summary_chunks(section)
            chunks += _prose_chunks(section)  # intro and rule boxes around the table
        elif section.number == FAQ_SECTION:
            chunks += _faq_chunks(section)
        else:
            chunks += _prose_chunks(section)
    ids = [c.chunk_id for c in chunks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate chunk ids: {sorted(duplicates)}")
    return chunks

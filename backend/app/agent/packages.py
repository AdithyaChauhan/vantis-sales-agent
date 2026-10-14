"""Reference packages, all or none (T13c).

In chat 1 of Adithya's break-it test the price answer named only HomePro 5 "for homes with higher electricity use",
the one package row the search had found. With no system size settled, a reply names every reference package for the
customer's kind of property, or none: these helpers give the model the whole list when part of it was found, and
remove a partial list from a reply.

Package names and kinds come from the KB Section 9 rows (KbHints.guidance), never from code. The bill-turn rule
(package_case, T13b) stays in app/agent/leads.py.
"""
import re

from app.rag.answer import split_sentences, without_tags

# A system size in the message ("5kW", "3 kW"): the customer settled the size, so one package may be named (T12b). A
# format check (a number with a kW unit), like the phone check, for the guards on our own reply (T13e audit: kept)
SIZE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:kw|kwp|kilowatts?)\b", re.IGNORECASE)
_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:\d+[.)]|[-*•])[ \t]+")  # "1. " or "- ": kept with the item, never a sentence
# A reply that already says the size is checked later (T13e)
_ASSESSMENT_RE = re.compile(r"\bassess|\bsite\s+(?:visit|survey|inspection)\b|\bfinal\s+size\b", re.IGNORECASE)


def reference_packages(guidance: list[dict]) -> dict[str, list[str]]:
    """{"home": [...], "business": [...]}: each KB Section 9 row's offering (its "...Offering: <name>" line), by the
    kind of customer the rest of the row names (residential: home; commercial or business: business)."""
    packages: dict[str, list[str]] = {}
    for row in guidance:
        name, rest = None, []
        for line in str(row.get("content", "")).split("\n"):
            key, _, value = line.partition(":")
            if "offering" in key.lower() and value.strip():
                name = value.strip()
            else:
                rest.append(line.lower())
        text = " ".join(rest)
        kind = "home" if "residential" in text else "business" if ("commercial" in text or "business" in text) else None
        if name and kind and name not in packages.get(kind, []):
            packages.setdefault(kind, []).append(name)
    return packages


def names_in(text: str, names: list[str]) -> list[str]:
    """The package names a text mentions ("Vantis HomePro 5" mentions HomePro 5), in the order of `names`."""
    found = []
    for name in names:
        pattern = r"\s+".join(re.escape(part) for part in name.split())
        if re.search(rf"\b{pattern}\b", text, re.IGNORECASE):
            found.append(name)
    return found


def size_settled(lead: dict, message: str) -> bool:
    """True once the customer gave a system size, in the draft or in this message: one package may then be named."""
    return bool(lead.get("system_size_kw") or SIZE_RE.search(message))


def add_all_packages(chunks: list[dict], guidance: list[dict], lead: dict, message: str) -> list[dict]:
    """When no size is settled and the found chunks name some reference packages of a kind but not all of them, the
    KB Section 9 rows not yet in the context are added, so the reply can name every package (or none)."""
    if not guidance or size_settled(lead, message):
        return chunks
    text = "\n".join(str(c.get("content", "")) for c in chunks)
    packages = reference_packages(guidance)
    if not any(0 < len(names_in(text, names)) < len(names) for names in packages.values()):
        return chunks
    present = {c["chunk_id"] for c in chunks}
    return chunks + [row for row in guidance if row["chunk_id"] not in present]


def all_or_no_packages(
    reply: str, packages: dict[str, list[str]], lead: dict, message: str, fallback: str
) -> tuple[str, list[str]]:
    """With no size settled, a reply that names some but not all of a kind's reference packages loses the sentences
    and list lines that name them (and a list intro left with no items), so it names none. A sentence that names a
    package the customer named in this message is always kept ("What is HomePro 5?", "HomeStart 3 vs HomePro 5?").
    `fallback` is the reply if nothing would be left. Returns (reply, the removed sentences)."""
    if not packages or size_settled(lead, message):
        return reply, []
    plain = without_tags(reply)
    partial = [n for names in packages.values() if 0 < len(names_in(plain, names)) < len(names) for n in names]
    if not partial:
        return reply, []
    asked = names_in(message, [n for names in packages.values() for n in names])
    original = reply.split("\n")
    lines, removed, emptied = [], [], []
    for line in original:
        marker = _LIST_MARKER_RE.match(line)
        prefix = marker.group(0) if marker else re.match(r"[ \t]*", line).group(0)
        kept, dropped_last = [], False
        for piece in split_sentences(line[len(prefix):]):
            text = without_tags(piece)
            if not text:  # tags alone follow the sentence before them
                if piece.strip() and not dropped_last:
                    kept.append(piece.strip())
                continue
            dropped_last = bool(names_in(text, partial)) and not names_in(text, asked)
            if dropped_last:
                removed.append(text)
            elif piece.strip():
                kept.append(piece.strip())
        lines.append(prefix + " ".join(kept) if kept else "")
        emptied.append(bool(line.strip()) and not kept)
    if not removed:
        return reply, []
    for i, line in enumerate(lines):  # "Here are the home packages:" whose items were all removed
        if line.rstrip().endswith(":"):
            block = []
            for j in range(i + 1, len(original)):
                if not original[j].strip():
                    break
                block.append(j)
            if block and all(emptied[j] for j in block):
                removed.append(without_tags(line).strip())
                lines[i] = ""
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text or fallback, removed



def add_size_line(reply: str, packages: dict[str, list[str]], line: str, tag: str = "") -> tuple[str, bool]:
    """On a turn whose message states a system size, a reply that names a reference package but never says the size
    is confirmed later gets `line` ("A site assessment confirms the final size…", with `tag`), before its closing
    question (T13e: Adithya's live test named HomePro 5 for "a 5kW system for my house" without it; KB Section 9 says
    not to assume 5 kW suits every household). Returns (reply, added)."""
    plain = without_tags(reply)
    if not any(names_in(plain, names) for names in packages.values()) or _ASSESSMENT_RE.search(plain):
        return reply, False
    sentence = f"{line.rstrip('.')} {tag}.".replace(" .", ".") if tag else line
    rows = [split_sentences(row) for row in reply.split("\n")]
    questions = [(i, j) for i, pieces in enumerate(rows) for j, piece in enumerate(pieces)
                 if without_tags(piece).endswith("?")]
    if questions and questions[-1] == (len(rows) - 1, len(rows[-1]) - 1):  # the reply ends with its question
        i, j = questions[-1]
        rows[i].insert(j, sentence)
    else:
        rows.append([sentence]) if not rows[-1] or rows[-1] == [""] else rows[-1].append(sentence)
    return "\n".join(" ".join(pieces) for pieces in rows), True


def _closing_question(reply: str) -> tuple[str, str]:
    """(the reply without its closing question, that question) when the reply ends with a question, else (reply, "")."""
    lines = reply.rstrip().split("\n")
    pieces = split_sentences(lines[-1])
    if pieces and without_tags(pieces[-1]).endswith("?"):
        rest = " ".join(pieces[:-1]).strip()
        return "\n".join(lines[:-1] + ([rest] if rest else [])).strip(), pieces[-1].strip()
    return reply.strip(), ""


def add_offerings_line(reply: str, catalog: str, intro: str, tag: str = "") -> tuple[str, bool]:
    """A reply to "what do you offer?" (the understand step's asks_offerings, T13e) that names none of the offerings
    in the KB Section 8 summary (`catalog`, KB data) gets that list, each offering with who it is for and the tag
    after the last one, before its closing question. Adithya's live test answered the assignment's first question
    from KB Section 1's list of services and named no package. Returns (reply, added)."""
    rows = [line[2:].split("; ") for line in catalog.split("\n") if line.startswith("- ")]
    names = [row[0] for row in rows] + [row[0].removeprefix("Vantis ") for row in rows]  # with or without the brand
    if not rows or names_in(without_tags(reply), names):
        return reply, False
    items = [f"{i}. {row[0]} ({row[1]})" if len(row) > 1 else f"{i}. {row[0]}" for i, row in enumerate(rows, start=1)]
    items[-1] = f"{items[-1]} {tag}".rstrip()
    body, question = _closing_question(reply)
    return "\n\n".join(part for part in (body, intro + "\n" + "\n".join(items), question) if part), True

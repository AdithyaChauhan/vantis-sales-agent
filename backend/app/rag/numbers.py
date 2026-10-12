"""The number guard (T12b): a reply never states a number, duration, amount, percentage or price that nothing
in the conversation gave.

KB Section 16: no invented price, savings, subsidy amount, installation date or warranty length. The KB gives
almost no numbers on purpose (package sizes, nothing else), so a number in a reply must come from this turn's
KB context, a tool result, the customer's own messages or the assistant's earlier (already checked) replies.
In the T12a slow-Google drill Groq wrote "the build phase takes a few weeks"; the prompts forbid it, this is
the safety net. Nothing here calls an LLM or the database.

What counts as a number:
- digits: "₹40,000", "25 years", "30%", "4.5 kW", "24/7" (an optional ₹/Rs before, k/thousand/lakh/crore after);
- number words with a unit: "two years", "twenty-five thousand";
- vague amounts with a unit: "a few weeks", "several months", "a year".
Number words without a unit ("one question", "a few details") do not count. KB tags, "KB Section 10" and list
markers ("1.") are ignored.
"""
import logging
import re
from dataclasses import dataclass, field

from app.leads.validators import AMOUNT_MULTIPLIERS
from app.rag.answer import split_sentences, without_tags

logger = logging.getLogger(__name__)

_WORD_VALUES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_DURATION = r"days?|weeks?|months?|years?|yrs?|hours?|hrs?|minutes?|mins?"
_MONEY = r"thousand|lakhs?|lacs?|crores?"
# Digits, with what tells the kind of number: a currency sign before, a multiplier and a unit after
_DIGITS_RE = re.compile(
    r"(?P<cur>₹\s*|\brs\.?\s*|\binr\s*)?(?<![\d.,])(?P<num>\d+(?:,\d+)*(?:\.\d+)?)(?!\d)"
    r"(?:\s*(?P<mult>k|thousand|lakhs?|lacs?|crores?|cr)\b)?"
    rf"(?:\s*-?\s*(?P<unit>%|(?:per\s*cent|percent|rupees?|{_DURATION})\b))?",
    re.IGNORECASE,
)
_NUMBER_WORD = "|".join(_WORD_VALUES)
_QUANTITY = rf"a\s+few|a\s+couple\s+of|couple\s+of|several|few|an?|(?:{_NUMBER_WORD})(?:[\s-]+(?:{_NUMBER_WORD}))*"
_WORDS_RE = re.compile(
    rf"\b(?P<qty>{_QUANTITY})\s+(?:more\s+)?(?P<unit>{_DURATION}|{_MONEY}|percent|per\s+cent)\b", re.IGNORECASE
)
# "next month", "this year", "every week" mean one of that unit (only read from the allowed texts)
_ONE_UNIT_RE = re.compile(r"\b(?:next|this|per|every|one)\s+(day|week|month|year)\b", re.IGNORECASE)
_RATE_BEFORE_RE = re.compile(r"(?:\d|\bk|\bthousand|\blakhs?|\bcrores?|\brupees?|\bunits?|\bkwh)\s*$", re.IGNORECASE)
_SECTION_RE =re.compile(r"\b(?:KB\s+)?Section\s+\d+\b", re.IGNORECASE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
_CONFIRMS_RE = re.compile(r"\bconfirm", re.IGNORECASE)
_UNIT_NAMES = {"yr": "year", "hr": "hour", "min": "minute", "lac": "lakh", "per cent": "percent"}
_VAGUE = {"a few": "few", "few": "few", "several": "several", "a couple of": "couple", "couple of": "couple"}


@dataclass(frozen=True)
class Mention:
    """One number in a text. kind: duration, money, percent or other (a count, a size, a phone number)."""

    text: str
    kind: str
    value: float | None = None  # digits and number words
    key: tuple[str, str] | None = None  # (quantity, unit) for "a few weeks" or "a month"
    digits: str = ""


@dataclass
class Known:
    """The numbers the allowed texts give: values, (quantity, unit) phrases and all their digits."""

    values: set[float] = field(default_factory=set)
    keys: set[tuple[str, str]] = field(default_factory=set)
    digits: str = ""


def _unit(word: str) -> str:
    """'Weeks' -> 'week', 'per cent' -> 'percent', 'lacs' -> 'lakh'."""
    word = " ".join(word.lower().split())
    if word != "percent" and word.endswith("s"):
        word = word[:-1]
    return _UNIT_NAMES.get(word, word)


def _kind(unit: str | None, money: bool) -> str:
    if unit == "percent" or unit == "%":
        return "percent"
    if money or unit in ("rupee", "thousand", "lakh", "crore"):
        return "money"
    if unit:
        return "duration"
    return "other"


def _number(text: str) -> float:
    return round(float(text.replace(",", "")), 4)


def _word_value(quantity: str) -> int:
    """'twenty-five' -> 25, 'two hundred' -> 200."""
    total = 0
    for word in re.split(r"[\s-]+", quantity.lower()):
        total = (total or 1) * 100 if word == "hundred" else total + _WORD_VALUES[word]
    return total


def _clean(text: str) -> str:
    """The text without KB tags, "KB Section N" and list markers, so their numbers are never read."""
    text = _SECTION_RE.sub(" ", without_tags(text))
    return "\n".join(_LIST_MARKER_RE.sub("", line) for line in text.split("\n"))


def number_mentions(text: str) -> list[Mention]:
    """Every number the text states, in order of the two kinds (digits, then words)."""
    text = _clean(text)
    mentions = []
    for m in _DIGITS_RE.finditer(text):
        value = _number(m["num"])
        mult = (m["mult"] or "").lower() or None
        if mult:
            value = round(value * AMOUNT_MULTIPLIERS.get(mult, 1), 4)
        unit = _unit(m["unit"]) if m["unit"] else None
        kind = _kind(unit, bool(m["cur"] or mult))
        mentions.append(Mention(m.group(0).strip(), kind, value=value, digits=re.sub(r"\D", "", m["num"])))
    for m in _WORDS_RE.finditer(text):
        quantity = " ".join(m["qty"].lower().split())
        unit = _unit(m["unit"])
        kind = _kind(unit, False)
        if quantity in _VAGUE:
            mentions.append(Mention(m.group(0), kind, key=(_VAGUE[quantity], unit)))
        elif quantity in ("a", "an"):
            if _RATE_BEFORE_RE.search(text[: m.start()]):  # "₹2,000 a month" is a rate, not a duration
                continue
            mentions.append(Mention(m.group(0), kind, key=("1", unit)))
        else:
            value = _word_value(quantity)
            if unit in ("thousand", "lakh", "crore"):
                value *= AMOUNT_MULTIPLIERS[unit]
            mentions.append(Mention(m.group(0), kind, value=float(value), key=(str(value), unit)))
    return mentions


def known_numbers(texts: list[str]) -> Known:
    """The numbers in the allowed texts. A "4k" gives both 4 and 4000, so "₹4,000" in a reply matches it."""
    known = Known()
    digit_runs = []
    for text in texts:
        cleaned = _clean(text)
        for m in _DIGITS_RE.finditer(cleaned):
            raw = _number(m["num"])
            known.values.add(raw)
            mult = (m["mult"] or "").lower()
            if mult:
                known.values.add(round(raw * AMOUNT_MULTIPLIERS.get(mult, 1), 4))
            if m["unit"]:
                known.keys.add((f"{raw:g}", _unit(m["unit"])))
        for mention in number_mentions(text):
            if mention.value is not None:
                known.values.add(mention.value)
            if mention.key:
                known.keys.add(mention.key)
        known.keys.update(("1", _unit(unit)) for unit in _ONE_UNIT_RE.findall(cleaned))
        digit_runs.append(re.sub(r"\D", "", text))
    known.digits = "|".join(digit_runs)
    return known


def is_supported(mention: Mention, known: Known) -> bool:
    """True if the allowed texts give this number: the same value, the same long digit run (a phone number
    echoed with spaces), or the same vague phrase and unit ("a few weeks")."""
    if mention.value is not None and mention.value in known.values:
        return True
    if len(mention.digits) >= 4 and mention.digits in known.digits:
        return True
    return mention.key is not None and mention.key in known.keys


def unsupported(text: str, known: Known) -> list[Mention]:
    """The numbers in `text` that the allowed texts do not give."""
    return [m for m in number_mentions(text) if not is_supported(m, known)]


def remove_unsupported_numbers(
    reply: str, allowed_texts: list[str], confirm_line: str
) -> tuple[str, list[str], list[str]]:
    """Remove every sentence that states a number the allowed texts don't give.
    Returns (reply, the removed numbers, the removed sentences without their KB tags).

    If a removed number was a duration, an amount or a percentage, `confirm_line` (the team confirms these after
    an assessment) takes the first removed sentence's place, unless the reply already says the team confirms
    something. A count such as "8 steps" is removed without it. The graph puts the removed sentences in the reply's
    steps (T13a), so each removal can be checked; the server log gets only their count.
    """
    known = known_numbers(allowed_texts)
    removed: list[Mention] = []
    sentences: list[str] = []
    marker_at = None  # (line index, piece index) of the first removed sentence
    lines = []
    for line in reply.split("\n"):
        marker = _LIST_MARKER_RE.match(line)
        prefix, body = (marker.group(0), line[marker.end():]) if marker else ("", line)
        kept, dropped_last = [], False
        for piece in split_sentences(body):
            bare = without_tags(piece)
            if not bare:  # tags alone follow the sentence before them
                if piece.strip() and not dropped_last:
                    kept.append(piece.strip())
                continue
            bad = unsupported(bare, known)
            dropped_last = bool(bad)
            if bad:
                removed += bad
                sentences.append(bare)
                if marker_at is None:
                    marker_at = (len(lines), len(kept))
                continue
            if piece.strip():
                kept.append(piece.strip())
        lines.append((prefix, kept, bool(body.strip())))
    if not removed:
        return reply, [], []
    text_lines = []
    for index, (prefix, kept, had_text) in enumerate(lines):
        if marker_at and marker_at[0] == index:
            kept = kept[: marker_at[1]] + ["\0"] + kept[marker_at[1]:]
        if kept:
            text_lines.append(prefix + " ".join(kept))
        elif not had_text:
            text_lines.append("")  # a blank line stays; a line whose sentences all went is dropped
    text = "\n".join(text_lines)
    needs_line = any(m.kind != "other" for m in removed) and not _CONFIRMS_RE.search(text.replace("\0", ""))
    text = text.replace("\0", confirm_line if needs_line else "")
    text = re.sub(r"[ \t]+\n", "\n", re.sub(r"(?<=\S) {2,}", " ", text))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # The count only: a removed sentence is reply text, which never goes into the server log
    logger.warning("reply stated a number no source gave; sentences removed", extra={"removed": len(sentences)})
    return text, [m.text for m in removed], sentences

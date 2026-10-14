"""Helpers for grounded answers: the context block, citations, sources and the retrieve step.

The agent (app/agent/graph.py) uses them; nothing here calls an LLM or the database.
"""
import logging
import re

from langchain_core.messages import HumanMessage

from app.core.prompts import load_prompt
from app.rag.keywords import key_words
from app.rag.retriever import RetrievedChunk, SearchResult

logger = logging.getLogger(__name__)

LLM_DOWN_REPLY = "I'm having trouble right now. Please try again in a minute."
SNIPPET_CHARS = 160
_CITATION_RE = re.compile(r"KB Section (\d+)")
_LABEL_RE = re.compile(r"([ \t]*)\[KB Section (\d+)(?::[^\[\]]*)?\]")  # spaces before it, section number
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # "4.5 kW" and "₹2,000" are not split
# A claim that something was saved or logged: "your details are saved", "I've recorded your bill". Future or
# conditional wording ("will be saved", "once you share it, I can save it") is not a claim (T12a). T13d: "noted",
# "made a note", "got your …" and "received your …" too ("I've noted that the house is in Lucknow" and "I've got your
# name, email, city and monthly bill" were said when nothing was stored); "got it" and "we've got three packages" are not
_SAVED = r"(?:saved|recorded|stored|registered|logged|noted)"
_GOT = r"(?:got|received|captured|taken\s+down)\s+(?:all\s+)?(?:of\s+)?your\b|(?:made|taken)\s+(?:a\s+)?note\b"
# Handing on is a claim only when the sentence names who got it: "I've passed your details to the team"
# (but "I've shared the steps above" is not). T13d: also without a subject ("Shared your details with the team"), and
# the team being notified, informed or alerted ("the team has been notified")
_HANDED_ON = r"(?:passed|forwarded|sent|shared|notified|informed|alerted)"
_DONE_BY = r"\b(?:i|we)(?:'ve|’ve| have)?\s+(?:now\s+|just\s+|also\s+|already\s+)?"
_DONE_TO = r"\b(?:is|are|was|were|has been|have been)\s+(?:now\s+|all\s+|already\s+|successfully\s+)?"
_SAVED_RE = re.compile(rf"{_DONE_BY}(?:{_SAVED}\b|{_GOT})|{_DONE_TO}{_SAVED}\b|^\s*(?:all\s+)?{_SAVED}\b",
                       re.IGNORECASE)
_HANDED_ON_RE = re.compile(rf"{_DONE_BY}{_HANDED_ON}\b|{_DONE_TO}{_HANDED_ON}\b|^\W*{_HANDED_ON}\b", re.IGNORECASE)
_TEAM_RE = re.compile(r"\b(?:team|sales|someone|representative|colleague|expert)s?\b", re.IGNORECASE)
# A thank-you is never a KB fact ("Thank you for your interest ... [KB Section 2]" in the T12a replay)
_THANKS_RE = re.compile(r"^\W*(?:thanks|thank you)\b", re.IGNORECASE)
# "The team will follow up" promises a human action that has not happened (KB Section 16); the assistant may
# only say the team can (T8b rule, slipped on Groq in the T12a replay). T13d: every human action, not only a call
# ("Vantis will arrange support" in Adithya's live chat). "will confirm" stays: our own callback texts say the team
# will confirm a logged request.
_PROMISE_RE = re.compile(
    r"(\s+will|'ll|’ll)(\s+(?:follow up|call|contact|reach out|get in touch|get back|arrange|send|visit|schedule|"
    r"book|assign|dispatch|fix|repair|replace|inspect|look into|take care of|sort out|handle|resolve|help|assist|"
    r"set up|connect|coordinate|organi[sz]e|provide|process|raise))\b", re.IGNORECASE
)
# KB Section 10's own words ("Reference Locations", "Reference service area") are office terms; the customer reads
# "service areas" (T12b). Only text outside [KB Section ...] tags is changed, so the label stays exact.
_TAG_OR_TEXT_RE = re.compile(r"(\[KB Section \d+(?::[^\[\]]*)?\])|([^\[]+|\[)")
_REFERENCE_AREA_RE = re.compile(r"\breference\s+(?:service\s+)?(?:locations?|areas?)\b", re.IGNORECASE)
# T13c: topics the KB never covers. A reply states no policy on them either way ("Vantis doesn't offer a specific
# discount rate" answered "can I get 20% discount?"); it says it can't confirm and offers the sales team
_UNCOVERED_RE = re.compile(r"\b(?:discounts?|coupons?|cashback|promo(?:tion)?s?|brands?|"
                           r"(?:special|current|seasonal|festive|festival|ongoing)\s+offers?)\b", re.IGNORECASE)
# T13d: the KB never says a Vantis service is free. "A free consultation" (Groq, live check of 13 Oct) loses the word;
# a sentence that says something is free, at no cost, complimentary or muft is replaced by the honest line
_FREE_WORDS = r"(?:free|complimentary|no-cost|muft)"
_SERVICE_NOUNS = (r"(?:consultations?|site\s+(?:assessments?|visits?|surveys?|inspections?)|assessments?|visits?|"
                  r"surveys?|inspections?|quotes?|quotations?|estimates?|installations?|maintenance|servicing|"
                  r"check-?ups?|checks?|calls?|advice|evaluations?|support|services?|demos?)")
_FREE_BEFORE_NOUN_RE = re.compile(rf"\b(?P<article>an?\s+)?(?<![-\w]){_FREE_WORDS}\s+(?=(?P<noun>{_SERVICE_NOUNS})\b)",
                                  re.IGNORECASE)
_FREE_CLAIM_RE = re.compile(
    r"\b(?:is|are|was|will\s+be|comes?|be)\s+(?:completely\s+|totally\s+|absolutely\s+|also\s+)?"
    r"(?:free|complimentary)\b(?![-\w])|"
    r"\bfree\s+of\s+(?:any\s+)?(?:cost|charge)\b|\bat\s+no\s+(?:extra\s+|additional\s+)?(?:cost|charge)\b|"
    r"\bfor\s+free\b|\bwithout\s+any\s+(?:extra\s+)?(?:cost|charge)\b|\bcosts?\s+(?:you\s+)?nothing\b|"
    r"\bmuft\b", re.IGNORECASE)
_KB_SAYS_FREE_RE = re.compile(r"\bfree\b(?![-\w])|\bcomplimentary\b|\bno\s+cost\b|\bwithout\s+charge\b", re.I)
# T13d: a reply that uses this turn's chunks but cites none gets the tag of the chunk it shares most key words with,
# only when at least this many are shared (so a source chip shows; Groq often leaves the tag out)
MIN_SHARED_FOR_TAG = 3
# A sentence that says what the assistant can't confirm or doesn't know (T13c; T13e: wider, and such a sentence carries
# a KB tag only when it also states a fact from that section, see strip_unsourced_citations)
_CANT_CONFIRM_RE = re.compile(
    r"\b(?:can(?:'|’)?t|cannot|can not|unable to|not able to)\s+(?:confirm|say|tell|guarantee|promise)\b|"
    r"\b(?:don(?:'|’)?t|do not)\s+have\s+(?:any\s+)?(?:specific\s+|exact\s+|more\s+)?"
    r"(?:information|details|info|data)\b|"
    r"\bnot sure\b|\bno\s+(?:specific\s+|exact\s+)?information\b|"
    r"\b(?:isn(?:'|’)?t|is not|aren(?:'|’)?t|are not)\s+(?:covered|mentioned|included|listed)\s+in\s+"
    r"the\s+information\b|"
    r"\binformation\s+I\s+have\s+(?:doesn(?:'|’)?t|does not)\b", re.IGNORECASE)
# T13e: the customer never reads "the knowledge base": "our current service list" in a sentence about places, else "the
# information I have" (Adithya's live test: "isn't listed in the knowledge base")
_KB_WORDS_RE = re.compile(r"\b(?:the|our|my|this)\s+(?:knowledge\s+base|KB)\b|\bknowledge\s+base\b", re.IGNORECASE)
_PLACE_WORDS_RE = re.compile(r"\b(?:cit(?:y|ies)|areas?|locations?|places?|towns?|regions?|serv(?:e|es|ed|ice)|"
                             r"cover(?:s|ed)?|listed)\b", re.IGNORECASE)
# T13e: a KB rule written for the assistant, never said to the customer ("No universal duration should be assumed")
_KB_RULE_RE = re.compile(
    r"\bshould\s+(?:not\s+)?be\s+(?:assumed|told|treated|promised|guaranteed|stated|invented|fabricated|recommended)\b|"
    r"\bno\s+universal\b|\b(?:do\s+not|don(?:'|’)?t)\s+assume\b|\bthe\s+AI\b|\bmust\s+not\b", re.IGNORECASE)
# Clause breaks inside a sentence; the break belongs to the clause after it
_CLAUSE_SPLIT_RE = re.compile(r"(,\s+(?:and|but|so|as|while|since|because)\s+|;\s+|\s+[–—-]\s+)", re.IGNORECASE)


def citation_label(chunk: RetrievedChunk) -> str:
    """The label the model must cite, e.g. [KB Section 8: Product & Service Catalog]."""
    return f"[KB Section {chunk.section_no}: {chunk.section_title}]"


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Each chunk as its label on one line and its text below, separated by blank lines."""
    return "\n\n".join(f"{citation_label(c)}\n{c.content}" for c in chunks)


def rag_message(
    question: str, chunks: list[RetrievedChunk], lead_progress: str = "", area_checks: str = "",
    package_guidance: str = "",
) -> HumanMessage:
    """The customer's message wrapped in the RAG prompt with the context (empty when nothing was found),
    (T13b) the package rule of a bill turn right after it (its line goes on other turns), the lead progress block
    (what is known and still needed; "(not tracked)" without the lead tools) and (T12b) the service-area checks
    the code made for the cities named ("(no city named)" when none).

    The customer's text is filled in last, so a "{context}" typed by the customer is left as it is.
    """
    text = load_prompt("rag_prompt").replace("{context}", format_context(chunks))
    text = text.replace("{package_guidance}\n\n", f"{package_guidance}\n\n" if package_guidance else "")
    text = text.replace("{lead_progress}", lead_progress or "(not tracked)")
    text = text.replace("{service_area_check}", area_checks or "(no city named)")
    return HumanMessage(text.replace("{question}", question))


def normalize_citations(reply: str) -> str:
    """Turn full-width brackets 【 】 into [ ] (gpt-oss on Groq sometimes cites with them)."""
    return reply.replace("【", "[").replace("】", "]")


def strip_tool_citations(reply: str, tool_names: list[str]) -> str:
    """Remove a tool cited like a source, e.g. "[schedule_callback result]" (only KB sections are sources).

    The system prompt forbids it; this is the safety net, and it logs a warning so a slipping prompt shows.
    """
    words = "|".join(re.escape(w) for w in ["result", *tool_names] if w)
    pattern = re.compile(rf"\s*\[(?!KB Section)[^\[\]]*\b(?:{words})\b[^\[\]]*\]", re.IGNORECASE)
    cleaned = pattern.sub("", reply)
    if cleaned != reply:
        logger.warning("reply cited a tool; removed")
    return cleaned


def split_sentences(line: str) -> list[str]:
    """One line of a reply as sentences; a piece holding only KB tags belongs to the sentence before it."""
    return _SENTENCE_SPLIT_RE.split(line)


def without_tags(text: str) -> str:
    """The text with its [KB Section N: Title] tags removed."""
    return _LABEL_RE.sub("", text).strip()


def is_claim(sentence: str) -> bool:
    """True if the sentence says something was saved, recorded or logged, or passed to the team (T12a)."""
    text = without_tags(sentence)
    return bool(_SAVED_RE.search(text) or (_HANDED_ON_RE.search(text) and _TEAM_RE.search(text)))


def is_honest(sentence: str) -> bool:
    """True for a sentence that says the assistant doesn't know or can't confirm something (T13e)."""
    return bool(_CANT_CONFIRM_RE.search(sentence))


def states_fact_from(sentence: str, section: int, chunks: list[RetrievedChunk]) -> bool:
    """True when the sentence, without its "I can't confirm" words, shares MIN_SHARED_FOR_TAG key words with that
    section's chunks: it also states a fact from that section (T13e)."""
    text = " ".join(c.content for c in chunks if c.section_no == section)
    return len(key_words(_CANT_CONFIRM_RE.sub(" ", sentence)) & key_words(text)) >= MIN_SHARED_FOR_TAG


def strip_unsourced_citations(reply: str, chunks: list[RetrievedChunk]) -> tuple[str, int]:
    """Remove the KB tags this turn's sources don't support (T12a): a section that is not among this turn's
    chunks (every tag when there are none, e.g. a validation message), and any tag on a question, a thank-you or
    a "saved" or "logged" sentence, which are never KB facts. Returns (reply, tags removed).

    T13d: tags standing alone after such a sentence ("…which option fits best? [KB Section 8: …]") belong to the facts
    before it, so they move to the end of the last factual sentence or list line (also on an earlier line) when their
    section is in this turn's chunks; they are removed only when no fact comes before them. The assignment's Q1 and
    Q2 (S01, S02 in the audit) lost their source chips because the model cited after its closing question.

    T13e: a sentence saying the assistant doesn't know or can't confirm something ("I don't have information on …")
    keeps a tag only when it also states a fact from that section (MIN_SHARED_FOR_TAG key words shared with its chunks,
    as in "Warranty depends on the equipment, but I can't confirm the exact duration"); otherwise its tags (inline or
    standing after it) are dropped, never moved (Adithya's live test: KB Section 1 on a timeline, KB Section 11 on
    cashback and on panel cleaning).

    The prompts ask for this; this is the safety net, and it logs a warning so a slipping prompt shows.
    """
    sections = {c.section_no for c in chunks}
    removed = 0

    def keep_tags(text: str, verdict: str, sentence: str = "") -> str:
        nonlocal removed

        def one(match: re.Match) -> str:
            nonlocal removed
            section = int(match.group(2))
            if section in sections and (verdict == "fact" or (verdict == "honest" and states_fact_from(
                    sentence, section, chunks))):
                return match.group(0)
            removed += 1
            return ""

        return _LABEL_RE.sub(one, text)

    lines: list[list[str]] = []
    indents: list[str] = []
    last_fact: tuple[int, int] | None = None  # (line, piece) of the last factual sentence so far
    moved = 0
    for line in reply.split("\n"):
        kept, verdict, sentence = [], "fact", ""
        for piece in split_sentences(line):
            bare = without_tags(piece)
            if bare:  # a sentence; tags alone take the verdict of the sentence before them
                sentence = bare
                verdict = ("other" if bare.endswith("?") or is_claim(bare) or _THANKS_RE.match(bare)
                           else "honest" if _CANT_CONFIRM_RE.search(bare) else "fact")
            elif verdict == "other" and last_fact is not None:  # tags alone after a question: back to the last fact
                tags = keep_tags(piece, "fact").strip()
                if tags:
                    i, j = last_fact
                    target = lines[i] if i < len(lines) else kept
                    target[j] = f"{target[j]} {tags}"
                    moved += 1
                continue
            piece = keep_tags(piece, verdict, sentence).strip()
            if piece:
                kept.append(piece)
                if bare and verdict == "fact":
                    last_fact = (len(lines), len(kept) - 1)
        indents.append(re.match(r"[ \t]*", line).group(0))  # keeps a list item's indentation
        lines.append(kept)
    if not removed and not moved:
        return reply, 0
    if removed:
        logger.warning("reply cited KB sections without support from this turn; removed", extra={"removed": removed})
    if moved:
        logger.info("KB tags after a question moved to the facts before it", extra={"moved": moved})
    text = "\n".join(indent + " ".join(kept) if kept else "" for indent, kept in zip(indents, lines))
    return text.strip(), removed


def soften_promises(reply: str) -> tuple[str, int]:
    """'our team will follow up' becomes 'our team can follow up' (T12a): a reply never promises that a person
    will call, contact or follow up. Returns (reply, changes). The system prompt asks for this; this is the net."""
    changed, count = _PROMISE_RE.subn(lambda m: " can" + m.group(2), reply)
    if count:
        logger.warning("reply promised a human action; softened", extra={"changed": count})
    return changed, count


def _uncovered_topics(sentence: str, chunks: list[RetrievedChunk]) -> list[str]:
    """The uncovered-topic words a sentence uses that no chunk of this turn mentions ("discount" with no KB text on
    discounts)."""
    found = []
    for match in _UNCOVERED_RE.finditer(sentence):
        word = r"\s+".join(re.escape(part) for part in match.group(0).lower().rstrip("s").split())
        if not any(re.search(rf"\b{word}s?\b", c.content, re.IGNORECASE) for c in chunks):
            found.append(match.group(0))
    return found


def remove_uncovered_policies(reply: str, chunks: list[RetrievedChunk], honest: str) -> tuple[str, list[str]]:
    """A sentence that states something about discounts, coupons, cashback, promotions, special offers or brands,
    when this turn's KB context never mentions that word, is replaced by `honest` (one line saying it can't be
    confirmed, with the sales team; T13c). Questions and sentences that already say it can't be confirmed stay.
    Returns (reply, the removed sentences). The system prompt asks for this; this is the net."""
    lines, removed, replaced = [], [], False
    for line in reply.split("\n"):
        kept, dropped_last = [], False
        for piece in split_sentences(line):
            text = without_tags(piece)
            if not text:  # tags alone follow the sentence before them
                if piece.strip() and not dropped_last:
                    kept.append(piece.strip())
                continue
            dropped_last = (not text.endswith("?") and not _CANT_CONFIRM_RE.search(text)
                            and bool(_uncovered_topics(text, chunks)))
            if dropped_last:
                removed.append(text)
                if not replaced:
                    kept.append(honest)
                    replaced = True
            elif piece.strip():
                kept.append(piece.strip())
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    if not removed:
        return reply, []
    logger.warning("reply stated a policy the KB doesn't cover; replaced", extra={"removed": len(removed)})
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), removed


def remove_free_claims(reply: str, chunks: list[RetrievedChunk], honest: str) -> tuple[str, list[str]]:
    """No claim that a Vantis service is free (T13d: the KB never says so; Groq wrote "a free consultation" twice in the
    live check of 13 Oct). "free", "complimentary" or "muft" before a service ("a free consultation") is dropped from
    the sentence; a sentence that says something is free, at no cost or costs nothing is replaced by `honest` once.
    Nothing changes when this turn's chunks say something is free. Questions stay. Returns (reply, the removed
    words or sentences)."""
    if any(_KB_SAYS_FREE_RE.search(c.content) for c in chunks):
        return reply, []
    removed: list[str] = []

    def drop_word(match: re.Match) -> str:
        removed.append(match.group(0).strip())
        noun = match.group("noun")
        if match.group("article"):  # "a free assessment" -> "an assessment"
            article = "an " if noun[0].lower() in "aeiou" else "a "
            return article[0].upper() + article[1:] if match.group("article")[0].isupper() else article
        return ""

    lines, replaced = [], False
    for line in reply.split("\n"):
        kept = []
        for piece in split_sentences(line):
            text = without_tags(piece)
            if text and not text.endswith("?") and _FREE_CLAIM_RE.search(text):
                removed.append(text)
                if not replaced:
                    kept.append(honest)
                    replaced = True
                continue
            if text and not text.endswith("?"):
                piece = _FREE_BEFORE_NOUN_RE.sub(drop_word, piece)
                piece = re.sub(r"^(\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), piece)
            if piece.strip():
                kept.append(piece.strip())
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    if not removed:
        return reply, []
    logger.warning("reply said a Vantis service is free; removed", extra={"removed": len(removed)})
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), removed


def add_missing_tag(reply: str, chunks: list[RetrievedChunk], skip: tuple[str, ...] = ()) -> tuple[str, str | None]:
    """When a reply cites no KB section but clearly uses this turn's chunks, add the tag of the chunk it shares the most
    key words with (at least MIN_SHARED_FOR_TAG), after the sentence that shares the most with it (T13d: Groq often
    leaves the tag out, so no source chip showed). Questions, claims, thank-yous and the fixed lines in `skip` are never
    tagged. Returns (reply, the tag added or None)."""
    if not chunks or _LABEL_RE.search(reply):
        return reply, None
    lines = [split_sentences(line) for line in reply.split("\n")]
    best = None  # (shared, line, piece, chunk)
    for i, pieces in enumerate(lines):
        for j, piece in enumerate(pieces):
            text = without_tags(piece)
            if (not text or text.endswith("?") or is_claim(text) or _THANKS_RE.match(text) or is_honest(text)
                    or any(text == fixed or text in fixed for fixed in skip)):
                continue
            words = key_words(text)
            for chunk in chunks:
                shared = len(words & key_words(chunk.content))
                if shared >= MIN_SHARED_FOR_TAG and (best is None or shared > best[0]):
                    best = (shared, i, j, chunk)
    if best is None:
        return reply, None
    _, i, j, chunk = best
    tag = citation_label(chunk)
    lines[i][j] = f"{lines[i][j].rstrip()} {tag}"
    logger.info("reply cited no KB section; added the best-matching one", extra={"section": chunk.section_no})
    return "\n".join(" ".join(pieces) for pieces in lines), tag


def plain_service_area_words(reply: str) -> tuple[str, int]:
    """'reference locations' and 'reference service area(s)' become 'service area(s)' outside KB tags (T12b).
    Returns (reply, changes). The system prompt asks for this; this is the net."""
    count = 0

    def plain(match: re.Match) -> str:
        nonlocal count
        count += 1
        word = match.group(0)
        plural = word.lower().endswith("s")
        text = "service areas" if plural else "service area"
        return text[0].upper() + text[1:] if word[0].isupper() else text

    def part(match: re.Match) -> str:
        return match.group(1) or _REFERENCE_AREA_RE.sub(plain, match.group(2))

    changed = _TAG_OR_TEXT_RE.sub(part, reply)
    if count:
        logger.warning("reply used KB office words for service areas; replaced", extra={"changed": count})
    return changed, count


def _sentences_outside_tags(reply: str, change) -> str:
    """The reply with `change(sentence_text, segment)` applied to each text segment outside KB tags, sentence by
    sentence (the sentence's own text, without tags, decides how it changes)."""
    lines = []
    for line in reply.split("\n"):
        pieces = []
        for piece in split_sentences(line):
            bare = without_tags(piece)
            pieces.append(_TAG_OR_TEXT_RE.sub(lambda m: m.group(1) or change(bare, m.group(2)), piece))
        lines.append(" ".join(pieces))
    return "\n".join(lines)


def plain_kb_words(reply: str) -> tuple[str, int]:
    """'the knowledge base' becomes 'our current service list' in a sentence about places and 'the information I have'
    otherwise, outside KB tags (T13e: the customer never reads our office words). Returns (reply, changes)."""
    count = 0

    def change(sentence: str, segment: str) -> str:
        def plain(match: re.Match) -> str:
            nonlocal count
            count += 1
            text = "our current service list" if _PLACE_WORDS_RE.search(sentence) else "the information I have"
            return text[0].upper() + text[1:] if match.group(0)[0].isupper() else text
        return _KB_WORDS_RE.sub(plain, segment)

    changed = _sentences_outside_tags(reply, change)
    if count:
        logger.warning("reply said 'knowledge base' to the customer; replaced", extra={"changed": count})
    return (changed, count) if count else (reply, 0)


def remove_kb_rules(reply: str) -> tuple[str, list[str]]:
    """A clause that quotes a KB rule written for the assistant ("no universal duration should be assumed", "do not
    assume 5 kW suits every household") is cut from its sentence; the fact clauses stay, with the sentence's KB tags at
    its end. A sentence that is all rule goes (T13e). Questions are left alone. Returns (reply, the removed clauses)."""
    removed: list[str] = []
    lines = []
    for line in reply.split("\n"):
        kept = []
        for piece in split_sentences(line):
            text = without_tags(piece)
            if not text or text.endswith("?") or not _KB_RULE_RE.search(text):
                if piece.strip():
                    kept.append(piece.strip())
                continue
            tags = " ".join(m.group(0).strip() for m in _LABEL_RE.finditer(piece))
            parts = _CLAUSE_SPLIT_RE.split(text)
            clauses, breaks = parts[0::2], [""] + parts[1::2]
            fact = ""
            for joint, clause in zip(breaks, clauses):
                if _KB_RULE_RE.search(clause):
                    removed.append(clause.strip(" .!"))
                else:
                    fact += (joint if fact else "") + clause
            fact = fact.strip()
            if not fact:
                continue
            end = text[-1] if text[-1] in ".!" else "."
            fact = fact[0].upper() + fact[1:].rstrip(" .!")
            kept.append(f"{fact} {tags}{end}" if tags else f"{fact}{end}")
        indent = re.match(r"[ \t]*", line).group(0)
        lines.append(indent + " ".join(kept) if kept else "")
    if not removed:
        return reply, []
    logger.warning("reply quoted a KB rule meant for the assistant; removed", extra={"removed": len(removed)})
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), removed


def tidy_citations(reply: str, chunks: list[RetrievedChunk]) -> str:
    """Each KB section cited once, with its full label (the prompt asks for this; this is the safety net).

    Only a section's last citation is kept (after a list, that is the end of the list), and a bare
    "[KB Section 10]" gets its title from the chunks so the chat page can show it as a tag.
    """
    titles = {c.section_no: c.section_title for c in chunks}
    remaining: dict[int, int] = {}
    for match in _LABEL_RE.finditer(reply):
        remaining[int(match.group(2))] = remaining.get(int(match.group(2)), 0) + 1

    def keep_last(match: re.Match) -> str:
        section = int(match.group(2))
        remaining[section] -= 1
        if remaining[section]:
            return ""  # a later citation of the same section stays
        title = titles.get(section)
        return match.group(1) + (f"[KB Section {section}: {title}]" if title else match.group(0).lstrip(" \t"))

    tidied = _LABEL_RE.sub(keep_last, reply)
    if tidied != reply:
        logger.info("citations tidied")
    return tidied


def cited_sections(reply: str) -> list[int]:
    """KB Section numbers the reply cites, in order of first mention, without repeats."""
    return list(dict.fromkeys(int(n) for n in _CITATION_RE.findall(reply)))


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The chunk text on one line, cut at a word boundary after at most `limit` characters."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"


def pick_sources(reply: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """One source per KB Section, because citations are per section.

    Uses the context chunks whose section the reply cites, in retrieval order; a reply that cites none (a
    question, a confirmation) has no sources (T12a: it used to get every context section, so a question showed
    a "KB 10" source). A section's first (highest-ranked) chunk gives the snippet.
    """
    cited = set(cited_sections(reply))
    used = [c for c in chunks if c.section_no in cited]
    if cited - {c.section_no for c in chunks}:
        logger.warning("reply cites a section that was not in the context")
    first_per_section = {}
    for c in used:
        first_per_section.setdefault(c.section_no, c)
    return [
        {
            "section": str(c.section_no),
            "title": c.section_title,
            "page": c.page,
            "snippet": snippet(c.content),
        }
        for c in first_per_section.values()
    ]


def retrieve_step(result: SearchResult) -> dict:
    """The retrieve entry of `steps`."""
    detail = f"{len(result.chunks)} chunks ({result.mode})"
    if result.no_context:
        detail = f"no relevant KB context ({result.mode})"
    return {
        "node": "retrieve",
        "detail": detail,
        "data": {"mode": result.mode, "chunks": len(result.chunks), "no_context": result.no_context},
    }

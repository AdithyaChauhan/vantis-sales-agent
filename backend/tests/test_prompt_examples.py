"""Prompts must not teach to the test: no quoted example may be close to an eval or live-test question."""
import re

import pytest
from rapidfuzz import fuzz

from app.core.config import BACKEND_DIR
from app.core.prompts import PROMPTS_DIR
from scripts.eval_retrieval import KB_QUESTIONS, OFFTOPIC_QUESTIONS, load_questions

MAX_SIMILARITY = 70  # rapidfuzz score (0-100); 70 or more counts as a paraphrase
QUOTED_RE = re.compile(r'"([^"]{3,})"')
LIVE_MESSAGE_RE = re.compile(r'ask\(client, "([^"]+)"')


def test_questions() -> list[str]:
    """Every eval question and every message the live tests send."""
    questions = [q.question for q in load_questions(KB_QUESTIONS) + load_questions(OFFTOPIC_QUESTIONS)]
    live = (BACKEND_DIR / "tests" / "test_live_chat.py").read_text(encoding="utf-8")
    return questions + LIVE_MESSAGE_RE.findall(live)


def prompt_examples() -> list[tuple[str, str]]:
    """(prompt file, quoted text) for every quoted example in every prompt."""
    return [
        (path.name, example)
        for path in sorted(PROMPTS_DIR.glob("*.md"))
        if path.name != "README.md"
        for example in QUOTED_RE.findall(path.read_text(encoding="utf-8"))
    ]


def similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b))


test_questions.__test__ = False  # a helper, not a test


def test_live_messages_are_found():
    live = [q for q in test_questions() if q in ("And in Kanpur?", "Yes please, connect me")]
    assert len(live) == 2


@pytest.mark.parametrize("prompt, example", prompt_examples())
def test_prompt_example_is_not_an_eval_question(prompt, example):
    closest = max(test_questions(), key=lambda q: similarity(example, q))
    score = similarity(example, closest)
    assert score < MAX_SIMILARITY, f'{prompt}: "{example}" is {score:.0f} similar to "{closest}"'

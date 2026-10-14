"""Prompts must not teach to the test: no quoted example may be close to an eval or live-test question."""
import re

import pytest
from rapidfuzz import fuzz

from app.core.config import BACKEND_DIR
from app.core.prompts import PROMPTS_DIR
from scripts.eval_retrieval import HOLDOUT_QUESTIONS, KB_QUESTIONS, OFFTOPIC_QUESTIONS, load_questions
from scripts.live_scenarios import EDGE, REPLAY, SLOW, T13D, T13E, T13E_FIX

MAX_SIMILARITY = 70  # rapidfuzz score (0-100); 70 or more counts as a paraphrase
QUOTED_RE = re.compile(r'"([^"]{3,})"')
LIVE_MESSAGE_RE = re.compile(r'ask\(client, "([^"]+)"')  # ask(client, "...") in the live tests
# Adithya's own test (T12 notes), replayed live in T12a; its one-word messages ("yes", "4pm") are generic
REPLAY_T12 = [
    "i want to install one at my organisation",
    "adithya, 329939",
    "naah. can you tell me the capital of jharkhand?",
    "is there any other info u need that can help u assist me apart from timing?",
    "are you sure? rooftop availability info is not needed?",
    "get back to me day after tomorrow",
]


def test_questions() -> list[str]:
    """Every eval question (T12b: the held-out ones too), every message the live tests and the live scenarios
    send and the replayed T12 conversation."""
    files = (KB_QUESTIONS, OFFTOPIC_QUESTIONS, HOLDOUT_QUESTIONS)
    questions = [q.question for path in files for q in load_questions(path)]
    questions += REPLAY_T12
    questions += [turn.message for scenario in EDGE + SLOW + REPLAY + T13D + T13E + T13E_FIX
                  for turn in scenario.turns]
    for path in sorted((BACKEND_DIR / "tests").glob("test_live_*.py")):  # every live test file
        questions += LIVE_MESSAGE_RE.findall(path.read_text(encoding="utf-8"))
    return questions


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
    found = set(test_questions())
    assert {"And in Kanpur?", "Yes please, connect me", "Rahul Sharma, 9876543210, rahul.sharma@example.com"} <= found
    assert {"rahul@gmail", "Do you also clean the water tank on the roof when installing solar?"} <= found


@pytest.mark.parametrize("prompt, example", prompt_examples())
def test_prompt_example_is_not_an_eval_question(prompt, example):
    closest = max(test_questions(), key=lambda q: similarity(example, q))
    score = similarity(example, closest)
    assert score < MAX_SIMILARITY, f'{prompt}: "{example}" is {score:.0f} similar to "{closest}"'

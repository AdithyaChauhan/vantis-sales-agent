"""Live edge-case scenarios and the held-out chat check (T12b), against a running serve_test_branch.

Dev only: every turn is a real chat turn (real LLM calls), so run it sparingly, after the backend tests and
`seed_demo_leads --reset` (the tests empty the test branch). Each scenario is one chat session.

Run:
    python -m scripts.serve_test_branch --port 8001                    (in another terminal)
    python -m scripts.live_scenarios --group holdout --base-url http://127.0.0.1:8001
    python -m scripts.live_scenarios --group edge --base-url http://127.0.0.1:8001
    python -m scripts.serve_test_branch --port 8001 --slow-gemini      (restart the server for the drill)
    python -m scripts.live_scenarios --group slow --base-url http://127.0.0.1:8001
    python -m scripts.live_scenarios --group replay --base-url http://127.0.0.1:8001   (T13c, about 25 model calls)
    python -m scripts.live_scenarios --group t13d --base-url http://127.0.0.1:8001     (T13d, about 25 model calls)

Groups:
- holdout: the 8 held-out questions (data/eval/holdout_eval.jsonl), one session each (assignment: unknown info)
- edge: missing information, invalid information, an existing customer, off-topic then back on topic
- slow: service failure; the server must run with --slow-gemini, and turns are 65 s apart so Groq's 8K tokens a
  minute can refill
- replay: C8, chat 1 of Adithya's break-it test on the live site (T13c), without its three retry turns, then the
  discount question
- t13d: the assignment's Q1 and Q2 keep their sources, "5 kW for my house" names its package, a what-if is not a fault,
  a Hindi business word, a call request is ready, "just exploring" is COLD, and no third ask after two skipped ones

For each turn it prints the reply and PASS or FAIL for each check, then a markdown table for docs/EVAL.md.
The number check reads the whole KB (read-only, from DATABASE_URL): a reply may state only numbers the KB, the
customer or an earlier reply gave. Exit code 1 if any check failed.
"""
import argparse
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.agent.welcome import REDIRECT_TEXT
from app.leads.validators import PHONE_INVALID
from app.rag.answer import LLM_DOWN_REPLY, is_claim, split_sentences, without_tags
from app.rag.numbers import known_numbers, number_mentions, unsupported

PAUSE_S = {"holdout": 20, "edge": 20, "slow": 65, "replay": 20, "t13d": 20}


@dataclass
class Context:
    """What a check may compare against: the KB, the customer's messages and the assistant's earlier replies."""

    kb_texts: list[str]
    customer: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)


Check = tuple[str, Callable[[dict, Context], bool]]


@dataclass
class Turn:
    message: str
    checks: list[Check]


@dataclass
class Scenario:
    id: str
    group: str
    title: str
    turns: list[Turn]


# ---------- checks (plain functions of the API response) ----------

def _reply(response: dict) -> str:
    return response.get("reply", "")


def _questions(reply: str) -> int:
    return sum(1 for line in reply.split("\n") for piece in split_sentences(line) if without_tags(piece).endswith("?"))


def _tool_details(response: dict, name: str) -> list[str]:
    return [s.get("detail", "") for s in response.get("steps", []) if s.get("node") == "tool" and s.get("name") == name]


def _lead(response: dict) -> dict:
    return response.get("lead") or {}


def no_invented_numbers() -> Check:
    """The app's number guard also allows tool results and validation messages; the script sees neither, so it
    allows the step details (e.g. "saved: request 4") and the phone validator's fixed message."""
    def check(response: dict, ctx: Context) -> bool:
        steps = [s.get("detail", "") for s in response.get("steps", [])]
        known = known_numbers(ctx.kb_texts + ctx.customer + ctx.replies + steps + [PHONE_INVALID, LLM_DOWN_REPLY])
        return not unsupported(_reply(response), known)
    return "no number the KB, the customer or an earlier reply didn't give", check


def no_amounts() -> Check:
    return ("no price, duration, amount or percentage",
            lambda r, c: not [m for m in number_mentions(_reply(r)) if m.kind != "other"])


def team_confirms() -> Check:
    pattern = re.compile(r"\b(?:confirm|verif|assess|consultation)", re.I)
    return "says the team confirms it", lambda r, c: bool(pattern.search(_reply(r)))


def at_most_one_question() -> Check:
    return "at most one question", lambda r, c: _questions(_reply(r)) <= 1


def exactly_one_question() -> Check:
    return "exactly one question", lambda r, c: _questions(_reply(r)) == 1


def plain_service_words() -> Check:
    pattern = re.compile(r"\breference\s+(?:service\s+)?(?:locations?|areas?)\b", re.I)
    return "no 'reference locations'", lambda r, c: not pattern.search(without_tags(_reply(r)))


def no_lead() -> Check:
    return "no lead saved", lambda r, c: not _lead(r).get("id")


def lead_saved() -> Check:
    return "lead saved", lambda r, c: bool(_lead(r).get("id"))


def no_saved_claim() -> Check:
    return ("no 'saved' claim",
            lambda r, c: not any(is_claim(p) for line in _reply(r).split("\n") for p in split_sentences(line)))


def known(name: str) -> Check:
    return f"{name} known", lambda r, c: name in _lead(r).get("known_fields", [])


def not_known(name: str) -> Check:
    return f"{name} not stored", lambda r, c: name not in _lead(r).get("known_fields", [])


def city_is(city: str) -> Check:
    return f"city is {city}", lambda r, c: _lead(r).get("city") == city


def mentions(pattern: str, label: str) -> Check:
    regex = re.compile(pattern, re.I)
    return label, lambda r, c: bool(regex.search(without_tags(_reply(r))))


def not_mentions(pattern: str, label: str) -> Check:
    regex = re.compile(pattern, re.I)
    return label, lambda r, c: not regex.search(without_tags(_reply(r)))


def cites(section: int) -> Check:
    return f"cites KB Section {section}", lambda r, c: f"[KB Section {section}:" in _reply(r)


def cites_any(*sections: int) -> Check:
    """At least one of `sections` is cited (T13b: KB Sections 8 and 9 both describe the packages)."""
    label = " or ".join(str(s) for s in sections)
    return f"cites KB Section {label}", lambda r, c: any(f"[KB Section {s}:" in _reply(r) for s in sections)


def names_every(names: list[str]) -> Check:
    """T13b: the reply names each of `names` (a bill turn lists the packages, it never picks one)."""
    return (f"names {', '.join(names)}",
            lambda r, c: all(name.lower() in without_tags(_reply(r)).lower() for name in names))


def city_checked(status: str = "listed") -> Check:
    """The city check said `status`: the code's check before the model (T12b) or the model's tool call."""
    def check(response: dict, ctx: Context) -> bool:
        by_code = [s.get("detail", "") for s in response.get("steps", []) if s.get("node") == "area_check"]
        return any(d.endswith(f": {status}") for d in by_code + _tool_details(response, "check_service_area"))
    return f"service area check: {status}", check


def city_checked_listed() -> Check:
    return city_checked("listed")


_CITY_QUESTION_RE = re.compile(r"\b(?:city|town|where|location|located)\b", re.I)


def no_city_question(city: str) -> Check:
    """T13a: no question sentence asks for the city or confirms `city` once the bot has it."""
    def check(response: dict, ctx: Context) -> bool:
        questions = [without_tags(p) for line in _reply(response).split("\n") for p in split_sentences(line)
                     if without_tags(p).endswith("?")]
        return not any(_CITY_QUESTION_RE.search(q) or city.lower() in q.lower() for q in questions)
    return "doesn't ask for the city again", check


# T13c: the words of a question that asks for each detail
_ASKS_FOR = {
    "city": r"\b(?:city|town|located|location|where)\b",
    "monthly_bill_inr": r"\bbill\b",
    "name": r"\bname\b",
    "phone": r"\b(?:phone|mobile|number)\b",
    "customer_type": r"\b(?:home|house)\s+or\s+(?:an?\s+|your\s+)?(?:small\s+)?(?:business|shop|office)\b|"
                     r"\bresidential\s+or\s+commercial\b|\bkind of (?:property|place)\b",
}
_HONEST_RE = re.compile(r"\b(?:can(?:'|’)?t|cannot|unable to)\s+confirm\b|\b(?:don(?:'|’)?t|do not)\s+have\s+"
                        r"(?:any\s+)?(?:information|details)\b|\bnot sure\b|\bteam can\b", re.I)


def _question_sentences(reply: str) -> list[str]:
    return [without_tags(p) for line in reply.split("\n") for p in split_sentences(line) if without_tags(p).endswith("?")]


def no_question_for_known() -> Check:
    """T13c: no question asks for a detail the lead already has (chat 1 asked "house or small business?" right after
    "the house is in Kanpur")."""
    def check(response: dict, ctx: Context) -> bool:
        known_fields = set(_lead(response).get("known_fields", []))
        return not any(field in known_fields and re.search(pattern, q, re.I)
                       for q in _question_sentences(_reply(response)) for field, pattern in _ASKS_FOR.items())
    return "asks for nothing it already has", check


def names_all_or_none(names: list[str]) -> Check:
    """T13c: the reply names every one of `names` or none of them (chat 1's price answer named only HomePro 5)."""
    def check(response: dict, ctx: Context) -> bool:
        named = [n for n in names if n.lower() in without_tags(_reply(response)).lower()]
        return len(named) in (0, len(names))
    return f"names all of {', '.join(names)} or none", check


def tool_ran_at_most_once(name: str) -> Check:
    return f"{name} ran at most once", lambda r, c: len(_tool_details(r, name)) <= 1


def callback_requested() -> Check:
    return ("callback request logged",
            lambda r, c: any(d in ("requested", "updated") for d in _tool_details(r, "schedule_callback"))
            or _lead(r).get("callback_status") == "requested")


def no_policy_on(word: str) -> Check:
    """T13c: every sentence about `word` (which the KB never mentions) is a question or says it can't be confirmed or
    that the team can tell; none states a policy either way."""
    def check(response: dict, ctx: Context) -> bool:
        sentences = [without_tags(p) for line in _reply(response).split("\n") for p in split_sentences(line)]
        return all(s.endswith("?") or _HONEST_RE.search(s) for s in sentences if word in s.lower())
    return f"states no {word} policy", check


def support_saved() -> Check:
    return ("support request saved, id in the reply",
            lambda r, c: any(d.startswith("saved") for d in _tool_details(r, "create_support_request"))
            and bool(re.search(r"\d", _reply(r))))


def is_redirect() -> Check:
    return "polite redirect", lambda r, c: _reply(r) == REDIRECT_TEXT or "only help" in _reply(r).lower()


def not_redirect() -> Check:
    return "a normal answer (not the redirect)", lambda r, c: _reply(r) != REDIRECT_TEXT


def not_down() -> Check:
    return "not 'try again'", lambda r, c: _reply(r) != LLM_DOWN_REPLY


def kb_grounded() -> Check:
    """T13d: the reply cites a KB section and the API returns its source."""
    return ("KB-grounded (a tag and a source)",
            lambda r, c: "[KB Section " in _reply(r) and bool(r.get("sources")))


def lead_status(status: str) -> Check:
    """The lead's status in the demo panel (serve_test_branch turns CHAT_SHOW_LEAD_SCORE on)."""
    return f"lead status {status}", lambda r, c: _lead(r).get("status") == status


def customer_type_is(kind: str) -> Check:
    return f"customer type {kind}", lambda r, c: _lead(r).get("customer_type") == kind


def not_a_support_case() -> Check:
    """T13d: a what-if is a question: no existing_customer_issue intent left, and no support request asked for."""
    def check(response: dict, ctx: Context) -> bool:
        understand = next((s for s in response.get("steps", []) if s.get("node") == "understand"), {})
        return "existing_customer_issue" not in understand.get("data", {}).get("intents", [])
    return "a what-if, not a support case", check


def understand_failed() -> Check:
    return ("understand failed (raw message used)",
            lambda r, c: any(s.get("node") == "understand" and s.get("detail") == "failed, raw message used"
                             for s in r.get("steps", [])))


def no_ask_once_parked() -> Check:
    """T13d: after two skipped questions about their details, the reply asks for none (passes while not parked)."""
    def check(response: dict, ctx: Context) -> bool:
        understand = next((s for s in response.get("steps", []) if s.get("node") == "understand"), {})
        if understand.get("data", {}).get("skipped_asks", 0) < 2:
            return True
        return not any(re.search(pattern, q, re.I) for q in _question_sentences(_reply(response))
                       for pattern in _ASKS_FOR.values())
    return "no question for details after two skipped ones", check


def answered_by_groq() -> Check:
    def check(response: dict, ctx: Context) -> bool:
        providers = [s["data"].get("provider") for s in response.get("steps", []) if s.get("node") == "agent"]
        return bool(providers) and all(p == "groq" for p in providers)
    return "answered by Groq (Gemini forced down)", check


# ---------- the scenarios ----------

COMMON = [no_invented_numbers(), at_most_one_question(), plain_service_words()]
UNKNOWN_NUMBER = [no_amounts(), team_confirms()]
NOT_FREE = not_mentions(r"\b(?:is|are|it's|its)\s+(?:completely\s+|totally\s+)?free\b|free of (?:cost|charge)|"
                        r"at no (?:extra )?cost", "doesn't say maintenance is free")


def holdout_scenarios(questions: list[tuple[str, str]]) -> list[Scenario]:
    """One session per held-out question, with its checks (h03/h04/h05/h08 ask for numbers the KB doesn't give)."""
    extra = {
        "h03": UNKNOWN_NUMBER,
        "h04": UNKNOWN_NUMBER + [NOT_FREE],
        "h05": UNKNOWN_NUMBER,
        "h06": [mentions(r"lucknow", "names Lucknow"), mentions(r"\bcover|\bserve|service area", "says it is covered"),
                cites(10)],
        "h07": [mentions(r"(?:don't|do not|doesn't|does not|not)\b.{0,40}\b"
                         r"(?:information|details|cover|mention|offer|sure)|can't (?:confirm|say)|not something",
                         "says the KB doesn't cover it"),
                mentions(r"\bteam\b|\bsales\b", "offers the team")],
        "h08": UNKNOWN_NUMBER,
    }
    return [Scenario(qid, "holdout", f"held-out {qid}", [Turn(question, COMMON + extra.get(qid, []))])
            for qid, question in questions]


NEVER_NOT_SERVED = not_mentions(
    r"\b(?:do not|don't|does not|doesn't|cannot|can't|won't)\s+(?:serve|install|cover|operate)",
    "never says Vantis doesn't serve it")
# KB Section 9's reference packages for a home: expected values, like PACKAGES in tests/test_live_chat.py (T13b)
HOME_PACKAGES = ["HomeStart 3", "HomePro 5", "HomeMax 10"]


def city_memory(sid: str, city: str, status: str) -> Scenario:
    """T13a: a city the customer only asked about counts as theirs and is never asked for again. T13b: the bill turn
    names every home package with a KB Section 8 or 9 tag (both describe the packages) instead of picking one."""
    return Scenario(sid, "edge", f"city memory: {city} asked about, then bill, then name and phone", [
        Turn(f"Do you install in {city}?", [city_checked(status), city_is(city), NEVER_NOT_SERVED, no_lead(), *COMMON]),
        Turn("My bill is about 4000 a month, it's for my home.",
             [no_city_question(city), city_is(city), known("monthly_bill_inr"), known("customer_type"),
              names_every(HOME_PACKAGES), cites_any(8, 9), no_lead(), *COMMON]),
        Turn("I'm Arjun, 9876543210", [no_city_question(city), lead_saved(), city_is(city), known("phone"), *COMMON]),
    ])


EDGE = [
    Scenario("C1", "edge", "missing information", [
        Turn("I want solar for my house", [exactly_one_question(), no_lead(), no_saved_claim(), no_invented_numbers()]),
    ]),
    Scenario("C3a", "edge", "invalid information: city typo, bill shorthand, bad phone, bad email", [
        # The unsaved draft keeps the customer's spelling; the lookup matches it and the saved lead says Delhi (turn 4)
        Turn("I want solar for my house in Dehli, my bill is about 4k a month",
             [city_checked_listed(), mentions(r"\bdelhi\b", "names Delhi"), known("monthly_bill_inr"), no_lead(),
              *COMMON]),
        Turn("Rahul Sharma, 329939",
             [not_known("phone"), mentions(r"phone|number|mobile", "asks for the phone again"), no_lead(),
              no_saved_claim(), *COMMON]),
        Turn("rahul@gmail",
             [not_known("email"), mentions(r"e-?mail", "asks for the email again"), no_lead(), no_saved_claim(),
              *COMMON]),
        Turn("9876500001", [lead_saved(), known("phone"), not_known("email"), city_is("Delhi"), *COMMON]),
    ]),
    Scenario("C3b", "edge", "invalid information: old city name", [
        Turn("Do you install in Gurgaon?", [city_checked_listed(), mentions(r"gurugram|gurgaon", "names the city"),
                                            cites(10), *COMMON]),
    ]),
    Scenario("C5", "edge", "existing customer", [
        Turn("my inverter shows a red light since Monday",
             [no_lead(), mentions(r"name|phone|contact|number", "asks who they are"), *COMMON]),
        Turn("Meera Iyer, 9876500003", [no_lead(), *COMMON]),
        Turn("It's a 3 kW rooftop system installed in 2025 in Noida", [support_saved(), no_lead(), *COMMON]),
    ]),
    Scenario("C6", "edge", "off-topic, then back on topic", [
        Turn("Who won the last cricket world cup?", [is_redirect(), no_lead()]),
        Turn("Okay. Do you install in Kanpur?", [not_redirect(), city_checked_listed(), cites(10), *COMMON]),
    ]),
    city_memory("C7", "Lucknow", "listed"),
    city_memory("C7b", "Raipur", "unverified"),
]

# T13c: chat 1 from Adithya's break-it test on the live site (13 Oct, 13:26-13:33 IST), without the three retry turns
# ("house", "home", "home" after model failures), then the discount question from chat 2
KNOWN = no_question_for_known()
REPLAY = [
    Scenario("C8", "replay", "chat 1 of the break-it test, then a discount", [
        Turn("solar lagwana hai ghar par. kitna kharcha hoga",
             [names_all_or_none(HOME_PACKAGES), no_amounts(), team_confirms(), KNOWN, no_lead(), *COMMON]),
        Turn("my bill is 6k",
             [known("monthly_bill_inr"), known("customer_type"), names_every(HOME_PACKAGES), cites_any(8, 9), KNOWN,
              no_lead(), *COMMON]),
        Turn("I'm Ravi, my number is 12345",
             [not_known("phone"), mentions(r"phone|number|mobile", "asks for the phone again"), no_lead(),
              no_saved_claim(), KNOWN, *COMMON]),
        Turn("sorry, 9812345678",
             [known("phone"), tool_ran_at_most_once("create_lead"), mentions(r"\bcity\b|\bwhere\b", "asks for the city"),
              no_lead(), no_saved_claim(), KNOWN, *COMMON]),
        Turn("actually the house is in Kanpur",
             [lead_saved(), city_is("Kanpur"), known("customer_type"), KNOWN, *COMMON]),
        Turn("call me tomorrow 5pm", [callback_requested(), known("intent_level"), KNOWN, *COMMON]),
        Turn("can I get 20% discount?",
             [no_policy_on("discount"), mentions(r"\bteam\b|\bsales\b", "offers the sales team"), KNOWN, *COMMON]),
    ]),
]

# T13d: Adithya's live findings and the assignment's own example questions
T13D = [
    Scenario("D1", "t13d", "assignment Q1 keeps its source", [
        Turn("What solar systems do you provide?", [not_down(), cites(8), kb_grounded(), *COMMON]),
    ]),
    Scenario("D2", "t13d", "assignment Q2 keeps its sources, no price", [
        Turn("How much does a 5kW system cost?", [not_down(), kb_grounded(), no_amounts(), *COMMON]),
    ]),
    Scenario("D3", "t13d", "5 kW for my house names its package", [
        Turn("I want a 5 kW system for my house", [mentions(r"HomePro\s*5", "names HomePro 5"), cites(8), *COMMON]),
    ]),
    Scenario("D4", "t13d", "a Hinglish what-if is a question, not a fault", [
        Turn("agar inverter kharab ho jaye to kya karna padega", [not_a_support_case(), not_down(), *COMMON]),
    ]),
    Scenario("D5", "t13d", "a Hindi business word", [
        Turn("meri dukan Lucknow mein hai, uske liye solar lagwana hai", [customer_type_is("commercial"), *COMMON]),
    ]),
    Scenario("D6", "t13d", "a call request is ready intent", [
        Turn("Can someone from Vantis call me?", [mentions(r"name|phone|number", "asks for name and phone"), *COMMON]),
        Turn("Neha Gupta, 9876500003, my house is in Noida",
             [lead_saved(), known("intent_level"), lead_status("WARM"), *COMMON]),
    ]),
    Scenario("D7", "t13d", "just exploring is COLD", [
        Turn("Just exploring solar for now, maybe next year", [no_lead(), *COMMON]),
        Turn("I'm Vikram Rao, 9876500004, house in Kanpur, bill around 3000",
             [lead_saved(), lead_status("COLD"), *COMMON]),
    ]),
    Scenario("D8", "t13d", "no third ask after two skipped questions", [
        Turn("I want rooftop solar for my home", [no_lead(), *COMMON]),
        Turn("What warranty do you give on the panels?", [no_ask_once_parked(), *COMMON]),
        Turn("And how long does an installation usually take?", [no_ask_once_parked(), *COMMON]),
    ]),
]

SLOW = [
    Scenario("C4", "slow", "service failure: Gemini forced down (--slow-gemini), lead saved on Groq", [
        Turn("Do you install in Kanpur? I'd like a 3 kW system for my house.",
             [not_down(), answered_by_groq(), city_checked_listed(), *COMMON]),
        Turn("I'd like it installed next month.", [not_down(), answered_by_groq(), no_lead(), *COMMON]),
        Turn("Rohan Verma, 9876500002",
             [not_down(), lead_saved(), city_is("Kanpur"), known("name"), known("phone"), *COMMON]),
    ]),
    # T13d (Adithya): understand fails in the drill (Groq is kept for answers), so a Hinglish question without a "?"
    # must still be searched on its raw words and answered from the KB
    Scenario("C4b", "slow", "a Hinglish question without '?' while understand fails", [
        Turn("subsidy milti hai kya sarkar se", [understand_failed(), not_down(), kb_grounded(), *COMMON]),
    ]),
]


# ---------- running ----------

@dataclass
class TurnResult:
    scenario: str
    turn: int
    message: str
    reply: str
    seconds: float
    failed: list[str]
    passed: list[str]
    llm_calls: int
    guards: list[str] = field(default_factory=list)  # each reply guard that fired, with what it removed (T13a)


def llm_calls_seen(response: dict) -> int:
    """Model calls visible in the steps: one per answered understand step and per agent step (a lower bound: failed
    attempts and the background reason are not in the steps)."""
    steps = response.get("steps", [])
    understood = sum(1 for s in steps if s.get("node") == "understand" and s.get("data", {}).get("provider"))
    return understood + sum(1 for s in steps if s.get("node") == "agent")


def run_scenario(scenario: Scenario, post: Callable[[dict], dict], kb_texts: list[str], pause_s: float,
                 sleep: Callable[[float], None] = time.sleep) -> list[TurnResult]:
    """Send the scenario's turns in one session and check every reply."""
    ctx = Context(kb_texts)
    session_id = None
    results = []
    for number, turn in enumerate(scenario.turns, start=1):
        if number > 1:
            sleep(pause_s)
        ctx.customer.append(turn.message)
        start = time.perf_counter()
        response = post({"message": turn.message, **({"session_id": session_id} if session_id else {})})
        seconds = time.perf_counter() - start
        session_id = response.get("session_id") or session_id
        failed, passed = [], []
        for name, check in turn.checks:
            try:
                ok = check(response, ctx)
            except Exception:
                ok = False
            (passed if ok else failed).append(name)
        results.append(TurnResult(scenario.id, number, turn.message, _reply(response), seconds, failed, passed,
                                  llm_calls_seen(response), guards_fired(response)))
        ctx.replies.append(_reply(response))
    return results


def guards_fired(response: dict) -> list[str]:
    """The reply guards that fired, e.g. 'removed a question about a city it already has: "Is it in Lucknow?"'."""
    fired = []
    for step in response.get("steps", []):
        if step.get("node") == "guard":
            removed = step.get("data", {}).get("removed")
            text = "; ".join(f'"{r}"' for r in removed) if isinstance(removed, list) else ""
            fired.append(f"{step.get('detail', '')}: {text}" if text else step.get("detail", ""))
    return fired


def print_turn(result: TurnResult) -> None:
    """One turn: the customer's message, the reply, the guards that fired and each check."""
    print(f"\n[{result.scenario} turn {result.turn}] {result.seconds:.1f} s")
    print(f"  customer: {result.message}")
    print(f"  reply:    {result.reply}")
    for guard in result.guards:
        print(f"    guard: {guard}")
    for name in result.passed:
        print(f"    PASS {name}")
    for name in result.failed:
        print(f"    FAIL {name}")


def markdown(results: list[TurnResult]) -> str:
    """The results as a table for docs/EVAL.md."""
    lines = ["| Scenario | Turn | Customer | Time | Result | Failed checks |", "|---|---|---|---|---|---|"]
    for r in results:
        verdict = "pass" if not r.failed else "FAIL"
        lines.append(f"| {r.scenario} | {r.turn} | {r.message} | {r.seconds:.1f} s | {verdict} | "
                     f"{'; '.join(r.failed) or '-'} |")
    return "\n".join(lines)


def load_kb_texts() -> list[str]:
    """Every searchable KB chunk's text, read-only from DATABASE_URL."""
    from sqlalchemy import text

    from app.core.db import get_engine

    engine = get_engine()
    if engine is None:
        raise SystemExit("DATABASE_URL is not set; the number check needs the KB.")
    with engine.connect() as conn:
        return [row[0] for row in conn.execute(text("SELECT content FROM kb_chunks WHERE chunk_type <> 'policy'"))]


def scenarios_for(group: str) -> list[Scenario]:
    if group == "holdout":
        from scripts.eval_retrieval import HOLDOUT_QUESTIONS, load_questions

        return holdout_scenarios([(q.id, q.question) for q in load_questions(HOLDOUT_QUESTIONS)])
    return {"edge": EDGE, "slow": SLOW, "replay": REPLAY, "t13d": T13D}[group]


def main(argv: list[str] | None = None) -> int:
    """Run one group against the server; print every reply, its checks and a markdown table."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Live T12b scenarios against a running serve_test_branch.")
    parser.add_argument("--group", choices=("holdout", "edge", "slow", "replay", "t13d"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--pause", type=float, default=None, help="seconds between turns (default: by group)")
    parser.add_argument("--only", default=None, help="run one scenario id (e.g. C3a)")
    args = parser.parse_args(argv)

    import httpx

    pause = PAUSE_S[args.group] if args.pause is None else args.pause
    kb_texts = load_kb_texts()
    client = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120)

    def post(body: dict) -> dict:
        response = client.post("/api/chat", json=body)
        if response.status_code != 200:
            return {"reply": f"HTTP {response.status_code}: {response.text[:200]}", "steps": []}
        return response.json()

    scenarios = [s for s in scenarios_for(args.group) if not args.only or s.id == args.only]
    if args.group == "slow":
        print("The server must run with --slow-gemini for this group.")
    results: list[TurnResult] = []
    for index, scenario in enumerate(scenarios):
        if index:
            time.sleep(pause)
        print(f"\n=== {scenario.id}: {scenario.title}")
        for result in run_scenario(scenario, post, kb_texts, pause):
            print_turn(result)
            results.append(result)
    failed = [r for r in results if r.failed]
    print(f"\n{len(results) - len(failed)} of {len(results)} turns passed every check; "
          f"model calls seen in the steps: {sum(r.llm_calls for r in results)} (a lower bound)\n")
    print(markdown(results))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

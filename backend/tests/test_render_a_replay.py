"""Replay of Adithya's live chat "Script A" (Render logs of 13 Oct, 12:00-12:04 UTC) under the T13d rules.

The log's 9 requests are messages 1, 2 and 4-10 (message 3, "kitna kharcha aayega pura", never reached the server). On
the day, one 503 paused gemini-3.5-flash-lite for 120 s for every use, a Groq token 429 paused Groq for 60 s, a 503 on
gemini-3.1-flash-lite paused it for 120 s; message 7 then failed in 1.35 s ("I'm having trouble"), and message 9 (name,
phone, email) failed after an 8 s timeout plus a Groq 429, so no lead was saved.

Here every message arrives at its logged time on a fake clock, and each model behaves as the log showed it:
- main (3.5-flash-lite): a 503 at 12:01:30, slow from 12:03:32 (Google answered after about 18 s, so the 8 s deadline
  timed out); times the log never tried it count as healthy;
- Groq: its 8K tokens a minute, with the logged token counts, so a 429 carries a real retry-after;
- fast (3.1-flash-lite): a 503 around 12:02:48.
Only the wait for a deadline is simulated (a slow model moves the clock by its deadline and times out); the skip list,
the turn's budget, the last try, the graph, the code-side save and the safety net are the real code. The second test
makes every model fail from message 7 on. The lead is written to the Neon test branch (skipped without it).
"""
import httpx
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import AgentDeps, build_graph
from app.agent.leads import LeadOps
from app.agent.state import turn_input
from app.agent.tools import make_tools
from app.agent.understand import KbHints
from app.core import llm
from app.core.llm import Candidate, SkipList, last_try_chain, model_chain
from app.leads import service
from app.rag.answer import LLM_DOWN_REPLY, split_sentences
from app.rag.retriever import RetrievedChunk, SearchResult
from app.services.service_areas import ServiceAreas
from tests.agent_fakes import tool_call, understood
from tests.test_answer import ReasonLLM

AREAS = ServiceAreas.from_rows([
    {"Region": "Uttar Pradesh", "Reference Locations": "Lucknow, Kanpur, Noida"},
    {"Region": "Delhi NCR", "Reference Locations": "Delhi, Noida"},
])


def kb(chunk_id, section_no, title, chunk_type, content):
    return {"chunk_id": chunk_id, "section_no": section_no, "section_title": title, "page": 4,
            "chunk_type": chunk_type, "content": content}


AREAS_SUMMARY = kb("s10-table-summary", 10, "Locations", "table_summary",
                   "All rows (Region; Reference Locations):\n- Uttar Pradesh; Lucknow, Kanpur, Noida\n"
                   "- Delhi NCR; Delhi, Noida")
CATALOG_SUMMARY = kb("s08-table-summary", 8, "Catalog", "table_summary",
                     "All rows (Offering; Primary Customer):\n- Plan A; Residential\n- Plan B; Residential")
GUIDANCE = [kb("s09-moderate", 9, "Guidance", "table_row",
               "Customer Signal: Moderate residential requirement\nPotential Reference Offering: Plan A"),
            kb("s09-higher", 9, "Guidance", "table_row",
               "Customer Signal: Higher residential requirement\nPotential Reference Offering: Plan B")]
HINTS = KbHints("KB Section 8: Catalog", {8: CATALOG_SUMMARY, 10: AREAS_SUMMARY}, GUIDANCE)
# The KB chunks a search finds, by a word of its query (synthetic text in the KB's shapes)
FOUND = {
    "subsidy": [kb("s15-subsidy", 15, "FAQ", "faq", "Q: Do you provide subsidy assistance?\n"
                   "A: The team can explain the considerations, but eligibility and amounts require verification.")],
    "instalment": [kb("s15-financing", 15, "FAQ", "faq", "Q: Do you provide financing?\n"
                      "A: Options may be available depending on the project. A representative can share more.")],
    "inverter": [kb("s11-inverter", 11, "Warranty Policy", "table_row",
                    "Component / Area: Inverter\nReference Policy: Warranty depends on the selected equipment.")],
    "warranty": [kb("s15-warranty", 15, "FAQ", "faq", "Q: What warranty do solar panels have?\n"
                    "A: Warranty coverage depends on the specific equipment and manufacturer terms.")],
    "chhat": [kb("s06-process", 6, "Installation", "prose", "A site assessment may evaluate rooftop availability, "
                 "installation area and property conditions.")],
    "solar": [kb("s02-segments", 2, "Segments", "prose", "Individual homeowners, independent houses and villas are "
                 "typical residential customers.")],
}


def search(query: str) -> SearchResult:
    for word, chunks in FOUND.items():
        if word in query.lower():
            return SearchResult(chunks=[RetrievedChunk(**c) for c in chunks], mode="hybrid")
    return SearchResult(chunks=[], mode="hybrid", no_context=True)


class World:
    """The fake clock (seconds since 12:00:00 UTC), the deadline of the call being made, and Groq's token use."""

    def __init__(self):
        self.now = 0.0
        self.deadline = None
        self.groq_used: list[tuple[float, int]] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class GoogleError(Exception):
    def __init__(self, code: int):
        super().__init__(f"{code} UNAVAILABLE")
        self.code = code


class GroqLimit(Exception):
    def __init__(self, retry_after: float):
        super().__init__("429 rate_limit_exceeded")
        self.status_code = 429
        self.response = httpx.Response(429, headers={"retry-after": f"{retry_after:.1f}"},
                                       request=httpx.Request("POST", "https://example.invalid"))


class Model:
    """One provider for one use. `health(t)` is "ok", "503" or "slow"; Groq also counts its tokens. When it answers, it
    returns the script's output for this turn: the understanding, or the next agent message."""

    LATENCY = {"understand": 2.5, "answer": 1.5}
    SLOW_S = 18  # how long Google took on 13 Oct (12:03:32 on)
    GROQ_TOKENS = {"understand": 2800, "answer": 3500}  # the logged sizes
    GROQ_PER_MINUTE = 8000

    def __init__(self, world, script, key, use, health=None):
        self.world, self.script, self.key, self.use, self.health = world, script, key, use, health
        self.calls = 0

    def _groq(self):
        now, need = self.world.now, self.GROQ_TOKENS[self.use]
        recent = [(t, n) for t, n in self.world.groq_used if t > now - 60]
        if sum(n for _, n in recent) + need > self.GROQ_PER_MINUTE:
            free_at = recent[0][0] + 60  # when the oldest use leaves the minute
            raise GroqLimit(max(1.0, free_at - now))
        self.world.groq_used.append((now, need))

    def invoke(self, messages, config=None):
        self.calls += 1
        state = self.health(self.world.now)
        if self.key.startswith("groq") and state == "ok":
            self._groq()
        if state == "503":
            self.world.now += 1
            raise GoogleError(503)
        if state == "slow" and self.world.deadline < self.SLOW_S:
            self.world.now += self.world.deadline
            raise llm.DeadlineExceeded(f"no answer within {self.world.deadline} s")
        self.world.now += self.SLOW_S if state == "slow" else self.LATENCY[self.use]
        turn = self.script[self.script["now"]]
        if self.use == "understand":
            return turn["understand"]
        return turn["answers"].pop(0)


def main_health(t):  # 503 at 12:01:30; slow from 12:03:32
    return "503" if 89 <= t < 100 else "slow" if t >= 212 else "ok"


def fast_health(t):  # 503 around 12:02:48
    return "503" if 166 <= t < 172 else "ok"


def groq_health(t):  # its limit is the token bucket
    return "ok"


def all_down_from_message_7(t):
    return "503" if t >= 166 else "ok"


MESSAGES = [  # (arrival in seconds after 12:00:00 UTC, message, understanding, the agent's messages if a model answers)
    (12, "hi mujhe ghar pe solar lagwana hai",
     understood(["greeting", "buying_interest"], "Customer Segments residential solar", customer_type="residential"),
     [tool_call("search_knowledge_base", {"query": "residential solar"}),
      AIMessage("Vantis helps homeowners with rooftop solar [KB Section 2: Segments].")]),
    (44, "mera bijli bill 3500 aata hai monthly, 3bhk hai, kaunsa system lena chahiye",
     understood(["question", "buying_interest", "gives_details"], None, monthly_bill_inr="3500",
                customer_type="residential"),
     [AIMessage("For a home: Plan A for moderate use, Plan B for higher use [KB Section 9: Guidance]; a site "
                "assessment decides which one fits.")]),
    (88.6, "subsidy milti hai kya sarkar se", understood(["question"], "subsidy eligibility"),
     [AIMessage("Eligibility and amounts need verification [KB Section 15: FAQ].")]),
    (113, "instalment me pay kar sakte hain", understood(["question"], "instalment financing"),
     [AIMessage("Options may be available depending on the project [KB Section 15: FAQ].")]),
    (140.7, "aap log noida me kaam karte ho",
     understood(["question"], "service locations", asked_city="noida", cities=["noida"]),
     [AIMessage("Vantis currently covers Noida [KB Section 10: Locations].")]),
    (167.7, "panel ki warranty kitne saal ki hoti hai", understood(["question"], "panel warranty"),
     [AIMessage("Warranty depends on the equipment and the manufacturer's terms [KB Section 15: FAQ].")]),
    (189.4, "agar inverter kharab ho jaye to kya karna padega", understood(["question"], "inverter warranty"),
     [AIMessage("Inverter warranty depends on the selected equipment [KB Section 11: Warranty Policy].")]),
    (212.3, "main Rakesh Sharma hu, mera number 9876543210 hai, email rakesh.sharma@example.com",
     understood(["gives_details"], None, name="Rakesh Sharma", phone="9876543210", email="rakesh.sharma@example.com"),
     [tool_call("create_lead", {}), AIMessage("Your details are saved, and the sales team can follow up.")]),
    (246, "mera ghar ki chhat pe thoda sa shade hai ped ka, kya problem hogi",
     understood(["question"], "rooftop shade site assessment"),
     [AIMessage("A site assessment checks the rooftop and shade [KB Section 6: Installation].")]),
]


def replay(leads_db, monkeypatch, main=main_health, fast=fast_health, groq=groq_health) -> tuple[list[dict], World]:
    world = World()
    script = {"now": 0}
    skips = SkipList(clock=world, sleep=world.sleep)

    def deadline(fn, seconds):  # only the wait is simulated: the fake moves the clock and times out itself
        world.deadline = seconds
        return fn()

    monkeypatch.setattr(llm, "call_with_deadline", deadline)
    health = {"gemini:main": main, "gemini:fast": fast, "groq:m": groq}

    def candidates(use, seconds_by_key, keys):
        return [Candidate(key, Model(world, script, key, use, health[key]), use, seconds_by_key[key]) for key in keys]

    understand_llm = model_chain(candidates("understand", {"gemini:main": 4, "gemini:fast": 4},
                                            ["gemini:main", "gemini:fast"]), skips)
    answers = candidates("answer", {"gemini:main": 8, "groq:m": 10, "gemini:fast": 8},
                         ["gemini:main", "groq:m", "gemini:fast"])
    agent_llm = model_chain(answers, skips)
    patient = Candidate("gemini:main", Model(world, script, "gemini:main", "answer", main), "answer", llm.LAST_TRY_S)
    agent_llm.last_try = last_try_chain(answers, patient, skips)
    ops = LeadOps(engine=leads_db, areas_fn=lambda: AREAS, reason_llm=ReasonLLM("A ready buyer."))
    tools = make_tools(search, lambda: AREAS, lambda: HINTS, ops)
    graph = build_graph(AgentDeps(understand_llm=understand_llm, agent_llm=agent_llm, final_llm=agent_llm,
                                  agent_llm_without_create=agent_llm, search_fn=search, areas_fn=lambda: AREAS,
                                  hints_fn=lambda: HINTS, tools=tools, ops=ops, checkpointer=InMemorySaver()))
    results = []
    for i, (arrival, message, understanding, agent_messages) in enumerate(MESSAGES):
        world.now = max(world.now, arrival)
        script["now"] = i
        script[i] = {"understand": understanding, "answers": list(agent_messages)}
        start = world.now
        with llm.turn_scope(clock=world):
            state = graph.invoke(turn_input(message, "script-a"), {"configurable": {"thread_id": "script-a"}})
        results.append({**state, "seconds": world.now - start, "message": message})
    return results, world


def test_script_a_under_the_t13d_rules_never_says_trouble_and_saves_the_lead(leads_db, monkeypatch):
    results, _ = replay(leads_db, monkeypatch)
    for turn in results:
        assert turn["reply"] != LLM_DOWN_REPLY, turn["message"]
        assert turn["seconds"] <= 40, (turn["message"], turn["seconds"])  # every turn ends within the turn's budget
    seventh = results[5]  # message 7, 12:02:47: failed in 1.35 s on the day
    assert seventh["answered_by"] == "model" and "Warranty" in seventh["reply"]
    ninth = results[7]  # message 9: name, phone and email
    assert ninth["lead_id"], ninth["reply"]
    lead = service.get_lead(leads_db, ninth["lead_id"])
    assert (lead.name, lead.phone, lead.email) == ("Rakesh Sharma", "+919876543210", "rakesh.sharma@example.com")
    assert (lead.city, lead.monthly_bill_inr, lead.customer_type) == ("Noida", 3500, "residential")  # 6 and 2
    # main's single 503 at 12:01:30 paused it for 15 s (it was 120 s, for every use), so main understood message 5
    # (12:01:53) and message 7 (12:02:47) itself
    assert results[3]["steps"][0]["data"]["provider"] == "gemini"
    assert results[5]["steps"][0]["detail"].startswith("intents")


def test_script_a_with_every_model_down_from_message_7_still_keeps_the_lead(leads_db, monkeypatch):
    results, _ = replay(leads_db, monkeypatch, main=all_down_from_message_7, fast=all_down_from_message_7,
                        groq=all_down_from_message_7)
    for turn in results[5:]:
        assert turn["reply"] != LLM_DOWN_REPLY, turn["message"]
        sentences = [s for line in turn["reply"].split("\n") for s in split_sentences(line) if s.strip()]
        assert any(not s.endswith("?") for s in sentences), turn["reply"]  # never only a question
        assert turn["seconds"] <= 40
    seventh, eighth, ninth, tenth = results[5:]
    assert seventh["reply"].startswith("Warranty coverage depends on the specific equipment and manufacturer terms. "
                                       "[KB Section 15: FAQ]")
    assert seventh["answered_by"] == "knowledge_base" and [s["section"] for s in seventh["sources"]] == ["15"]
    assert eighth["reply"].startswith("Inverter: Warranty depends on the selected equipment. "
                                      "[KB Section 11: Warranty Policy]")
    # message 9: understand failed too, so the name ("main … hu"), phone and email were read in code, and finalize
    # saved the lead with the city from message 6's question and the need from message 2
    assert ninth["steps"][0]["data"]["from_text"] == ["email", "name", "phone"]
    lead = service.get_lead(leads_db, ninth["lead_id"])
    assert (lead.name, lead.city, lead.monthly_bill_inr) == ("Rakesh Sharma", "Noida", 3500)
    assert any(s["node"] == "save" for s in ninth["steps"]) and "saved" in ninth["reply"]
    assert tenth["reply"].startswith("A site assessment may evaluate rooftop availability")

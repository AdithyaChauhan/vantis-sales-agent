"""T13c: reference packages, all or none. Chat 1's price answer named only HomePro 5 "for homes with higher electricity
use". Package names come from fake KB Section 9 rows shaped like the real ones. Fake models and search only."""
import pytest

from app.agent.packages import add_all_packages, all_or_no_packages, names_in, reference_packages, size_settled
from app.agent.welcome import PACKAGES_FIT_TEXT
from app.rag.retriever import RetrievedChunk, SearchResult
from tests.agent_fakes import GUIDANCE, FakeSearch, Scripted, answer, make_graph, run, understood


def row(chunk_id: str, signal: str, offering: str) -> dict:
    """A KB Section 9 row as kb_chunks stores it."""
    return {"chunk_id": chunk_id, "section_no": 9, "section_title": "Reference Package Guidance", "page": 3,
            "chunk_type": "table_row",
            "content": f"Customer Signal: {signal}\nPotential Reference Offering: {offering}\n"
                       "Do Not Assume: Do not assume final system size."}


ROWS = [
    row("s09-moderate", "Moderate residential requirement", "HomeStart 3"),
    row("s09-higher", "Higher residential requirement", "HomePro 5"),
    row("s09-large", "Large residential requirement", "HomeMax 10"),
    row("s09-small-commercial", "Small commercial requirement", "Business 15"),
    row("s09-larger-business", "Larger small-business requirement", "Business 25"),
]
PACKAGES = reference_packages(ROWS)
HOMEPRO_ROW = {"chunk_id": "s08-homepro", "section_no": 8, "section_title": "Catalog", "page": 3,
               "chunk_type": "table_row", "content": "Offering: Vantis HomePro 5\nPrimary Customer: Residential"}


def guard(reply: str, message: str = "How much would solar cost for my home?", lead: dict | None = None):
    return all_or_no_packages(reply, PACKAGES, lead or {}, message, PACKAGES_FIT_TEXT)


def test_the_packages_and_their_kind_come_from_the_kb_rows():
    assert PACKAGES == {"home": ["HomeStart 3", "HomePro 5", "HomeMax 10"], "business": ["Business 15", "Business 25"]}
    assert reference_packages(GUIDANCE) == {"home": ["Plan A", "Plan B"]}  # the agent fakes' rows
    assert reference_packages([]) == {}


def test_names_in_finds_a_name_inside_the_catalog_name():
    assert names_in("The Vantis HomePro 5 is for higher use.", PACKAGES["home"]) == ["HomePro 5"]
    assert names_in("HomePro 50 and homestart  3", PACKAGES["home"]) == ["HomeStart 3"]


def test_size_settled():
    assert size_settled({"system_size_kw": 5.0}, "hi") and size_settled({}, "I want a 5 kW system")
    assert not size_settled({"monthly_bill_inr": 6000}, "my bill is 6k")


def test_one_home_package_with_no_size_is_removed():
    # chat 1, turn 1 (the price question)
    reply = ("Prices depend on your property and a site assessment. For homes with higher electricity use, the "
             "HomePro 5 is a reference option [KB Section 9: Reference Package Guidance]. Would you like a callback?")
    kept, removed = guard(reply)
    assert kept == "Prices depend on your property and a site assessment. Would you like a callback?"
    assert removed == ["For homes with higher electricity use, the HomePro 5 is a reference option."]


def test_every_home_package_is_kept():
    reply = ("For a home there are three reference packages:\n1. HomeStart 3: moderate use\n2. HomePro 5: higher use\n"
             "3. HomeMax 10: large homes [KB Section 9: Reference Package Guidance]\nA site assessment decides.")
    assert guard(reply) == (reply, [])


def test_a_partial_list_and_its_intro_go():
    reply = "Here are two options:\n1. HomeStart 3: moderate use\n2. HomePro 5: higher use\n\nWould you like a callback?"
    kept, removed = guard(reply)
    assert kept == "Would you like a callback?"
    assert removed == ["HomeStart 3: moderate use", "HomePro 5: higher use", "Here are two options:"]


def test_a_reply_with_nothing_left_gets_the_fallback():
    assert guard("HomePro 5 suits higher use.")[0] == PACKAGES_FIT_TEXT


@pytest.mark.parametrize("lead, message", [
    ({"system_size_kw": 5.0}, "How much would it cost?"),  # a size in the draft
    ({}, "How much is a 5 kW system?"),  # a size in this message
])
def test_a_settled_size_keeps_the_one_package(lead, message):
    reply = "The HomePro 5 is the 5 kW reference package [KB Section 8: Catalog]."
    assert guard(reply, message, lead) == (reply, [])


@pytest.mark.parametrize("message, reply", [
    ("what is HomePro 5?", "HomePro 5 is a rooftop solution for higher household use [KB Section 8: Catalog]."),
    ("HomeStart 3 vs HomePro 5?",
     "HomeStart 3 is for moderate household use, while HomePro 5 is for higher use [KB Section 8: Catalog]."),
])
def test_a_package_the_customer_named_is_never_removed(message, reply):
    # Adithya (planning): the guard never removes a sentence about a package named in this message
    assert guard(reply, message) == (reply, [])


def test_only_the_packages_the_customer_did_not_name_go():
    reply = "HomePro 5 is for higher household use. If your use is moderate, HomeStart 3 may also fit."
    assert guard(reply, "what is HomePro 5?") == ("HomePro 5 is for higher household use.",
                                                   ["If your use is moderate, HomeStart 3 may also fit."])


def test_every_package_of_both_kinds_is_kept():
    reply = "Homes: HomeStart 3, HomePro 5 and HomeMax 10. Businesses: Business 15 and Business 25."
    assert guard(reply, "What packages do you have?") == (reply, [])


# ---------- the whole list joins the context when part of it was found ----------

def test_part_of_the_list_brings_every_section_9_row():
    chunks = add_all_packages([HOMEPRO_ROW], ROWS, {}, "how much does solar cost for my home?")
    assert [c["chunk_id"] for c in chunks] == ["s08-homepro"] + [r["chunk_id"] for r in ROWS]


@pytest.mark.parametrize("chunks, lead, message", [
    ([], {}, "how much?"),  # nothing found
    ([HOMEPRO_ROW], {"system_size_kw": 5.0}, "how much?"),  # a settled size
    ([HOMEPRO_ROW], {}, "how much is 5kW?"),
    (ROWS[:3], {}, "what do you offer for homes?"),  # every home package already there
])
def test_no_rows_added(chunks, lead, message):
    assert add_all_packages(chunks, ROWS, lead, message) == chunks


# ---------- through the graph (fake rows: Plan A and Plan B for a home) ----------

PLAN_A_ROW = RetrievedChunk("s08-plan-a", 8, "Catalog", 3, "table_row", "Offering: Plan A, 3 kW")


def test_the_price_question_gets_every_row_and_a_partial_reply_is_cut():
    agent = Scripted(answer("It depends on your home. Plan A suits moderate use [KB Section 8: Catalog]. "
                            "Would you like a callback?"))
    search = FakeSearch(SearchResult(chunks=[PLAN_A_ROW], mode="hybrid"))
    graph, _ = make_graph(Scripted(understood(["question"], "Pricing residential")), agent, search=search)
    state = run(graph, "How much does solar cost for my home?")
    assert [c["chunk_id"] for c in state["chunks"]] == ["s08-plan-a", "s09-moderate", "s09-higher"]
    retrieve = next(s for s in state["steps"] if s["node"] == "retrieve")
    assert retrieve["data"]["added_package_rows"] == 2
    assert state["reply"] == "It depends on your home. Would you like a callback?"
    [step] = [s for s in state["steps"] if s["node"] == "guard"]
    assert step["detail"] == "removed a partial list of reference packages"
    assert step["data"]["removed"] == ["Plan A suits moderate use."]


def test_the_whole_list_passes_the_graph():
    reply = "For a home: Plan A for moderate use and Plan B for higher use [KB Section 9: Guidance]."
    graph, _ = make_graph(Scripted(understood(["question"], "Pricing residential")), Scripted(answer(reply)),
                          search=FakeSearch(SearchResult(chunks=[PLAN_A_ROW], mode="hybrid")))
    state = run(graph, "How much does solar cost for my home?")
    assert state["reply"] == reply and not [s for s in state["steps"] if s["node"] == "guard"]

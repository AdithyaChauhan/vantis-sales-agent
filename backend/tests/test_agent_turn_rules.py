"""T13d turn rules through the real graph with scripted models (no network, no database): a what-if is not a fault, a
look-alike place is never a listed city (a locality is), a stated size brings the catalog, Hindi place words, and no
more than one more ask after a skipped question."""
from app.agent.leads import SKIPPED_LINE, LeadOps
from app.rag.answer import is_claim, soften_promises
from app.rag.retriever import RetrievedChunk, SearchResult
from tests.agent_fakes import CATALOG_SUMMARY, FakeSearch, Scripted, answer, make_graph, run, understood
from tests.leads_data import AREAS
from tests.test_agent_leads import lead_status_text

NO_DB = LeadOps(engine=None, areas_fn=lambda: AREAS)  # lead tools on, nothing gets ready to save in these tests


def test_a_what_if_is_a_question_not_a_fault():
    """Adithya's live chat: "agar inverter kharab ho jaye to kya karna padega" opened a support case."""
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(["existing_customer_issue"], None, issue="inverter kharab")),
                          Scripted(answer("Warranty depends on the equipment.")), search=search)
    state = run(graph, "agar inverter kharab ho jaye to kya karna padega")
    assert state["support_case"] is False and "existing_customer_issue" not in state["intents"]
    assert search.queries == ["agar inverter kharab ho jaye to kya karna padega"]  # searched as a question
    assert state["steps"][0]["data"]["what_if"] is True and not state["support"].get("issue")


def test_a_problem_the_customer_has_now_still_opens_the_support_case():
    graph, _ = make_graph(Scripted(understood(["existing_customer_issue"], None, issue="inverter stopped")),
                          Scripted(answer("Sorry to hear that.")))
    state = run(graph, "My inverter stopped working, what if it's not under warranty?")
    assert state["support_case"] is True and state["steps"][0]["data"]["what_if"] is False


def test_a_landmark_phrase_is_never_stored_as_the_listed_city():
    """Adithya: "Lucknow ke paas ek gaon" is a village near Lucknow, unverified, whatever the extractor wrote."""
    graph, _ = make_graph(Scripted(understood(["buying_interest", "gives_details"], None, city="Lucknow")),
                          Scripted(answer("Thanks.")))
    state = run(graph, "Lucknow ke paas ek gaon mein mera ghar hai")
    assert state["lead"]["city"] == "Lucknow ke paas"
    assert [(c["city"], c["status"]) for c in state["area_checks"]] == [("Lucknow ke paas", "unverified")]


def test_a_locality_of_a_listed_city_stays_that_city():
    graph, _ = make_graph(Scripted(understood(["buying_interest", "gives_details"], None, city="Lucknow")),
                          Scripted(answer("Thanks.")))
    state = run(graph, "My house is in Gomti Nagar, Lucknow")
    assert state["lead"]["city"] == "Lucknow" and state["area_checks"][0]["status"] == "listed"
    graph, _ = make_graph(Scripted(understood(["buying_interest", "gives_details"], None,
                                              city="Gomti Nagar, Lucknow")), Scripted(answer("Thanks.")))
    assert run(graph, "My house is in Gomti Nagar, Lucknow")["area_checks"][0]["status"] == "listed"


def test_i_want_a_5_kw_system_gets_the_catalog_without_a_search():
    """The assignment's example: details only, so no KB search ran, and the reply couldn't name the 5 kW package."""
    search = FakeSearch()
    agent = Scripted(answer("Plan B is the 5 kW reference package [KB Section 8: Catalog]; a site assessment "
                            "confirms the size."))
    graph, _ = make_graph(Scripted(understood(["buying_interest", "gives_details"], None, system_size_kw="5 kW",
                                              property_type="house")), agent, search=search)
    state = run(graph, "I want a 5 kW system for my house")
    assert search.queries == [] and state["chunks"] == [CATALOG_SUMMARY]
    assert "Plan B 5 kW" in str(agent.calls[0][-1].content)
    assert "[KB Section 8: Catalog]" in state["reply"] and [s["section"] for s in state["sources"]] == ["8"]
    assert state["steps"][0]["data"]["catalog_for_size"] is True


def test_hindi_business_words_count_as_a_business():
    """Adithya's live chat: "dukan" (shop) gave no kind of place."""
    graph, _ = make_graph(Scripted(understood(["buying_interest"], None)), Scripted(answer("Thanks.")))
    state = run(graph, "meri dukan ke liye solar lagwana hai")
    assert state["lead"]["customer_type"] == "commercial" and state["steps"][0]["data"]["from_words"] == [
        "customer_type"]


def asks_graph(agent_replies, understandings):
    """(graph, the scripted agent) with the lead tools on."""
    agent = Scripted(*[answer(t) for t in agent_replies])
    graph, _ = make_graph(Scripted(*understandings), agent, ops=NO_DB)
    return graph, agent


def test_after_one_skipped_question_it_asks_once_more_then_stops_asking():
    """Adithya's live chat: the city was asked 4 times while the customer kept asking other things."""
    graph, agent = asks_graph(
        ["Great choice! Which city is your home in?",
         "Warranty depends on the equipment. Which city is the property in?",
         "Timing varies with the site. Which city are you in?",
         "Sure! Which city is the property in?"],
        [understood(["buying_interest"], None, customer_type="residential"),
         understood(["question"], "warranty"), understood(["question"], "installation time"),
         understood(["buying_interest"], None)])
    run(graph, "I want solar for my home", thread="skip")
    second = run(graph, "what warranty do you give?", thread="skip")
    assert second["skipped_asks"] == 1 and second["reply"].endswith("Which city is the property in?")  # one more ask
    third = run(graph, "how long does installation take?", thread="skip")
    assert third["skipped_asks"] == 2
    assert third["reply"] == "Timing varies with the site."  # the third ask is removed
    assert SKIPPED_LINE in lead_status_text(agent, 2)
    assert any(s["node"] == "guard" and s["data"].get("removed") == ["Which city are you in?"]
               for s in third["steps"])
    fourth = run(graph, "ok I want to go ahead", thread="skip")  # a clear buying signal: asking may start again
    assert fourth["skipped_asks"] == 0 and fourth["reply"].endswith("Which city is the property in?")


def test_giving_any_detail_or_asking_to_be_contacted_starts_the_count_again():
    graph, _ = asks_graph(
        ["Which city is your home in?", "Which city is it in?", "Thanks! What's your name?"],
        [understood(["buying_interest"], None, customer_type="residential"),
         understood(["question"], "warranty"),
         understood(["gives_details"], None, city="Kanpur")])
    run(graph, "I want solar for my home", thread="count")
    assert run(graph, "what warranty?", thread="count")["skipped_asks"] == 1
    assert run(graph, "Kanpur", thread="count")["skipped_asks"] == 0


def test_claims_and_promises_from_the_live_chats():
    assert is_claim("I've noted that the house is in Lucknow.") and is_claim("The team has been notified.")
    assert not is_claim("Got it, thanks!") and not is_claim("We've got three home packages.")
    assert soften_promises("Vantis will arrange support for your inverter.")[0] == (
        "Vantis can arrange support for your inverter.")
    assert soften_promises("I've logged it; the team will confirm it.")[1] == 0  # our own callback words stay


def test_a_call_request_read_in_code_when_understand_fails_is_ready_intent():
    """Live check of 13 Oct: Google timed out on every understand call, so "can someone call me?" lost its intent."""
    graph, _ = make_graph(Scripted(RuntimeError("main and fast down")), Scripted(answer("Sure! Your name and phone?")),
                          ops=NO_DB)
    state = run(graph, "Can someone from Vantis call me?")
    assert "contact_request" in state["intents"] and state["lead"]["intent_level"] == "ready"



def test_a_question_about_the_offer_brings_the_catalog_with_or_without_understand():
    """Round 2: on 13 Oct the raw search for the assignment's Q1 found only Section 1."""
    for understand_llm in (Scripted(understood(["question"], "Company Overview")), Scripted(RuntimeError("down"))):
        agent = Scripted(answer("We offer Plan A and Plan B [KB Section 8: Catalog]."))
        graph, _ = make_graph(understand_llm, agent)
        state = run(graph, "What solar systems do you provide?")
        assert state["chunks"][0] == CATALOG_SUMMARY and state["steps"][0]["data"]["catalog_for_offer"] is True
        assert "Plan B 5 kW" in str(agent.calls[0][-1].content)


def test_a_reply_without_a_tag_gets_one_and_its_source_chip():
    """Round 2: the drill's C4b reply used the subsidy chunks but cited none, so no source chip showed."""
    subsidies = RetrievedChunk("s13-subsidies", 13, "Subsidies", 4, "prose",
                               "Government subsidy programs, eligibility requirements, and amounts may change over time.")
    agent = Scripted(answer("Government subsidies exist, but eligibility and the amount need checking for you. "
                            "Would you like the team to help?"))
    graph, _ = make_graph(Scripted(understood(["question"], "subsidy")), agent,
                          search=FakeSearch(SearchResult(chunks=[subsidies], mode="hybrid")))
    state = run(graph, "subsidy milti hai kya sarkar se")
    assert "checking for you. [KB Section 13: Subsidies] Would you like" in state["reply"]
    assert [s["section"] for s in state["sources"]] == ["13"]
    assert any(s["detail"] == "added the KB tag the reply left out" for s in state["steps"])

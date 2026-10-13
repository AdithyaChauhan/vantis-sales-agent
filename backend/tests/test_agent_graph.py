"""The agent graph with fake models (no network): routing, tools, memory, failures and small-to-big."""
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import welcome
from app.agent.graph import MAX_TOOL_ROUNDS, add_table_summaries
from app.rag.answer import LLM_DOWN_REPLY, snippet
from app.rag.retriever import RetrievedChunk, SearchResult
from tests.agent_fakes import (
    AREAS_SUMMARY,
    CATALOG_ROW,
    CATALOG_SUMMARY,
    FAQ,
    FakeSearch,
    Scripted,
    answer,
    make_graph,
    run,
    tool_call,
    understood,
)

# T13c: both fake reference packages (tests/agent_fakes.py GUIDANCE), since a reply with one of them and no
# settled size loses it (all or none)
CITED = "We offer Plan A and Plan B [KB Section 8: Catalog]."


def nodes(state) -> list[str]:
    return [s["node"] if s["node"] != "tool" else f"tool:{s['name']}" for s in state["steps"]]


def test_greeting_gets_the_welcome_without_an_agent_call():
    agent = Scripted(answer("should not be called"))
    graph, search = make_graph(Scripted(understood(["greeting"])), agent)
    state = run(graph, "hi")
    assert state["reply"] == welcome.WELCOME_TEXT
    assert agent.calls == [] and search.queries == []
    assert nodes(state) == ["understand", "reply"] and state["sources"] == []


def test_thanks_gets_the_closing_line():
    graph, _ = make_graph(Scripted(understood(["thanks_or_bye"])), Scripted(answer("no")))
    state = run(graph, "thanks!")
    assert state["reply"] == welcome.CLOSING_TEXT
    assert state["reply"] not in (welcome.WELCOME_TEXT, welcome.REDIRECT_TEXT)


def test_off_topic_gets_the_redirect():
    agent = Scripted(answer("no"))
    graph, _ = make_graph(Scripted(understood(["off_topic"])), agent)
    assert run(graph, "tell me a joke")["reply"] == welcome.REDIRECT_TEXT
    assert agent.calls == []


def test_mixed_intents_go_to_the_agent():
    agent = Scripted(answer(CITED))
    graph, search = make_graph(
        Scripted(understood(["thanks_or_bye", "question"], "installation in Delhi")), agent
    )
    state = run(graph, "thanks, and do you install in Delhi?")
    assert state["reply"] == CITED
    assert search.queries == ["installation in Delhi"] and len(agent.calls) == 1


def test_question_path_with_sources_and_steps():
    agent = Scripted(answer(CITED))
    graph, search = make_graph(Scripted(understood(["question"], "Product & Service Catalog offerings")), agent)
    state = run(graph, "What solar systems do you provide?")
    assert search.queries == ["Product & Service Catalog offerings"]
    assert state["reply"] == CITED
    assert [s["section"] for s in state["sources"]] == ["8"]
    assert nodes(state) == ["understand", "retrieve", "agent"]
    assert state["steps"][2]["data"]["provider"] == "gemini"
    # the agent got the system prompt, then the current message wrapped in the RAG prompt with the context
    sent = agent.calls[0]
    assert "[KB Section 8: Catalog]\nOffering: Plan A" in sent[-1].content
    assert "What solar systems do you provide?" in sent[-1].content
    assert state["degraded"] is False


def test_short_reply_goes_to_the_agent_with_the_history():
    offer = "Yes, you can get a quote [KB Section 15: FAQ]. Would you like me to connect you with the Vantis sales team?"
    agent = Scripted(answer(offer), answer("You can share your details."))
    understand_llm = Scripted(
        understood(["question"], "quote"),
        understood(["reply_to_assistant"]),
    )
    graph, _ = make_graph(understand_llm, agent)
    run(graph, "Can I get a quote?")
    state = run(graph, "yes please")
    assert state["reply"] == "You can share your details."
    assert nodes(state) == ["understand", "agent"]  # no search query: straight to the agent, never the redirect
    second_call = agent.calls[1]
    texts = [str(m.content) for m in second_call]
    assert any("Can I get a quote?" == t for t in texts)  # earlier customer message
    assert any("connect you with the Vantis sales team" in t for t in texts)  # the earlier sales offer
    assert "yes please" in texts[-1]


def test_tool_loop_with_the_service_area_check():
    agent = Scripted(
        tool_call("check_service_area", {"city": "Pune"}),
        answer("Pune is not in our listed service areas; the team can check [KB Section 10: Locations]."),
    )
    graph, _ = make_graph(Scripted(understood(["question"], "installation in Pune")), agent)
    state = run(graph, "Can you install solar panels in Pune?")
    assert nodes(state) == ["understand", "retrieve", "agent", "tool:check_service_area", "agent"]
    assert state["steps"][3]["detail"] == "Pune: unverified"
    tool_result = agent.calls[1][-1]
    assert isinstance(tool_result, ToolMessage) and '"status": "unverified"' in tool_result.content
    assert [s["section"] for s in state["sources"]] == ["10"]  # the tool's KB Section 10 chunk


def test_city_answer_cites_the_service_area_summary_not_a_retrieved_row():
    delhi_row = RetrievedChunk("s10-delhi-ncr", 10, "Locations", 3, "table_row",
                               "Region: Delhi NCR Reference Locations: Delhi, Gurugram")
    agent = Scripted(
        tool_call("check_service_area", {"city": "Karimnagar"}),
        answer("Service in Karimnagar isn't confirmed yet; the team can check [KB Section 10: Locations]."),
    )
    understand = Scripted(understood(["question"], "service area Karimnagar", asked_city="Karimnagar"))
    graph, _ = make_graph(understand, agent, search=FakeSearch(SearchResult(chunks=[delhi_row, FAQ])))
    state = run(graph, "can you install it in Karimnagar?")
    [area] = [s for s in state["sources"] if s["section"] == "10"]
    assert area["snippet"] == snippet(AREAS_SUMMARY["content"])  # the tool's summary chunk, not the Delhi row


def test_a_tool_cited_like_a_source_is_removed_from_the_reply():
    graph, _ = make_graph(Scripted(understood(["question"], "x")),
                          Scripted(answer("Logged; the team will confirm it [schedule_callback result].")))
    assert run(graph, "call me at 5")["reply"] == "Logged; the team will confirm it."


def test_listed_city_and_the_search_tool():
    agent = Scripted(
        tool_call("check_service_area", {"city": "kanpur"}, "c1"),
        tool_call("search_knowledge_base", {"query": "installation process"}, "c2"),
        answer("Kanpur is listed [KB Section 10: Locations]."),
    )
    search = FakeSearch()
    graph, _ = make_graph(Scripted(understood(["question"])), agent, search=search)
    state = run(graph, "Do you install in Kanpur?")
    details = [s["detail"] for s in state["steps"] if s["node"] == "tool"]
    assert details == ["kanpur: listed", "2 chunks: KB Sections 8, 15"]
    # T12b: a question always searches; with no query from the model, the message itself is the query
    assert search.queries == ["Do you install in Kanpur?", "installation process"]


def test_tool_rounds_are_limited():
    final = Scripted(answer("Here is what I found."))
    agent = Scripted(tool_call("search_knowledge_base", {"query": "x"}))  # always asks for a tool
    graph, _ = make_graph(Scripted(understood(["question"], "x")), agent, final_llm=final)
    state = run(graph, "loop forever")
    assert len(agent.calls) == MAX_TOOL_ROUNDS
    assert len(final.calls) == 1
    assert state["reply"] == "Here is what I found."
    assert state["tool_rounds"] == MAX_TOOL_ROUNDS


def test_understand_failure_uses_the_raw_message():
    agent = Scripted(answer(CITED))
    graph, search = make_graph(Scripted(RuntimeError("quota")), agent)
    state = run(graph, "What do you offer?")
    assert search.queries == ["What do you offer?"]
    assert state["reply"] == CITED
    assert state["degraded"] is True
    assert state["steps"][0]["detail"] == "failed, raw message used"


def test_unparsable_understanding_also_falls_back():
    broken = {"raw": AIMessage(content="?"), "parsed": None, "parsing_error": ValueError("bad json")}
    graph, search = make_graph(Scripted(broken), Scripted(answer(CITED)))
    state = run(graph, "What do you offer?")
    assert search.queries == ["What do you offer?"] and state["degraded"] is True


def test_agent_failure_is_degraded():
    graph, _ = make_graph(Scripted(understood(["question"], "q")), Scripted(RuntimeError("both down")))
    state = run(graph, "What do you offer?")
    assert (state["reply"], state["degraded"], state["sources"]) == (LLM_DOWN_REPLY, True, [])


def test_empty_agent_reply_is_degraded():
    graph, _ = make_graph(Scripted(understood(["question"], "q")), Scripted(answer("   ")))
    state = run(graph, "What do you offer?")
    assert (state["reply"], state["degraded"]) == (LLM_DOWN_REPLY, True)


def test_no_context_gives_the_agent_an_empty_context():
    search = FakeSearch(SearchResult(chunks=[CATALOG_ROW], mode="hybrid", no_context=True))
    agent = Scripted(answer("I don't have that information."))
    graph, _ = make_graph(Scripted(understood(["question"], "weather")), agent, search=search)
    state = run(graph, "weather?")
    assert state["chunks"] == [] and state["no_context"] is True
    assert "<context>\n\n</context>" in agent.calls[0][-1].content
    assert state["sources"] == []


def test_memory_keeps_only_customer_messages_and_replies():
    agent = Scripted(tool_call("check_service_area", {"city": "Delhi"}), answer("Delhi is listed."), answer("Kanpur too."))
    understand_llm = Scripted(understood(["question"], "Delhi"), understood(["question"], "Kanpur"))
    graph, _ = make_graph(understand_llm, agent)
    run(graph, "Do you provide installation in Delhi?")
    state = run(graph, "And in Kanpur?")
    assert [type(m).__name__ for m in state["messages"]] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
    assert state["messages"][1].content == "Delhi is listed."
    # turn 2's understand prompt holds turn 1; per-turn fields were reset
    second_prompt = understand_llm.calls[1][0].content
    assert "Customer: Do you provide installation in Delhi?" in second_prompt
    assert "Assistant: Delhi is listed." in second_prompt
    assert nodes(state) == ["understand", "retrieve", "agent"]
    assert state["tool_rounds"] == 0
    assert all(not isinstance(m, ToolMessage) for m in state["scratch"])


def test_separate_sessions_do_not_share_memory():
    understand_llm = Scripted(understood(["question"], "q"))
    graph, _ = make_graph(understand_llm, Scripted(answer("ok")))
    run(graph, "first", thread="a")
    run(graph, "second", thread="b")
    assert "Customer: first" not in understand_llm.calls[1][0].content
    assert "(no earlier messages)" in understand_llm.calls[1][0].content


def test_lead_details_merge_and_errors_stay_per_turn():
    understand_llm = Scripted(
        understood(["buying_interest"], None, city="Lucknow", phone="12345", monthly_bill_inr="4k"),
        understood(["reply_to_assistant"], None, name="Rahul Sharma"),
    )
    graph, _ = make_graph(understand_llm, Scripted(answer("Thanks.")))
    first = run(graph, "Lucknow, bill 4k, call 12345")
    # T13c: buying interest with no intent from the extractor counts as "interested"
    assert first["lead"] == {"city": "Lucknow", "monthly_bill_inr": 4000, "intent_level": "interested"}
    assert set(first["validation_errors"]) == {"phone"}
    second = run(graph, "I'm Rahul Sharma")
    assert second["lead"] == {"city": "Lucknow", "monthly_bill_inr": 4000, "intent_level": "interested",
                              "name": "Rahul Sharma"}
    assert second["validation_errors"] == {}
    assert second["steps"][0]["data"]["details"] == ["name"]


def test_small_to_big_is_off_by_default_and_adds_the_summary_when_on():
    off, _ = make_graph(Scripted(understood(["question"], "catalog")), Scripted(answer(CITED)))
    state = run(off, "What do you offer?")
    # the 2 found, plus (T13c) the KB Section 9 rows, since the found row names only one of the two packages
    assert len(state["chunks"]) == 4 and state["steps"][1]["data"]["added_package_rows"] == 2
    on, _ = make_graph(Scripted(understood(["question"], "catalog")), Scripted(answer(CITED)), small_to_big=True)
    state = run(on, "What do you offer?")
    assert [c["chunk_id"] for c in state["chunks"]] == ["s08-plan-a", "s15-quote", "s08-table-summary"]
    assert state["steps"][1]["data"]["added_summaries"] == 1


def test_add_table_summaries_rules():
    row, faq = CATALOG_ROW.__dict__, FAQ.__dict__
    summaries = {8: CATALOG_SUMMARY}
    assert add_table_summaries([faq], summaries) == [faq]  # no catalog row: nothing added
    assert add_table_summaries([row, CATALOG_SUMMARY], summaries) == [row, CATALOG_SUMMARY]  # already there
    assert add_table_summaries([row], {}) == [row]  # no summary in the DB


def test_logs_hold_no_message_text(caplog):
    graph, _ = make_graph(Scripted(RuntimeError("down")), Scripted(RuntimeError("down")))
    with caplog.at_level(logging.DEBUG, logger="app"):
        run(graph, "My number is 9876543210, secret question")
    logged = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert "9876543210" not in logged and "secret question" not in logged


def test_a_kb_tag_outside_this_turns_sources_is_removed():
    # Problem 4 (T12a): KB tags only on facts from this turn's sources (here KB Sections 8 and 15)
    reply = ("We offer Plan A and Plan B [KB Section 8: Catalog]. Panels follow the maker's warranty "
             "[KB Section 11: Warranty].")
    graph, _ = make_graph(Scripted(understood(["question"], "Catalog")), Scripted(answer(reply)))
    state = run(graph, "What do you offer?")
    assert state["reply"] == "We offer Plan A and Plan B [KB Section 8: Catalog]. Panels follow the maker's warranty."
    assert [s["section"] for s in state["sources"]] == ["8"]
    assert any(s["node"] == "guard" and s["data"]["removed"] == 1 for s in state["steps"])


def test_a_turn_with_no_sources_has_no_kb_tags():
    # Problem 4: "KB 10" on the phone check, "KB 1" on a callback question
    reply = ("Thanks! That number looks short [KB Section 10: Locations]. "
             "Could you share your 10-digit mobile number? [KB Section 1: Overview]")
    graph, search = make_graph(Scripted(understood(["gives_details"], None, name="Asha Rao", phone="329939")),
                               Scripted(answer(reply)))
    state = run(graph, "asha, 329939")
    assert search.queries == []  # details only: no KB search, so no sources this turn
    assert state["reply"] == "Thanks! That number looks short. Could you share your 10-digit mobile number?"
    assert state["sources"] == []


def test_a_kb_tag_on_a_question_is_removed_even_when_the_section_was_found():
    reply = "We offer Plan A and Plan B [KB Section 8: Catalog]. Would you like a quote [KB Section 15: FAQ]?"
    graph, _ = make_graph(Scripted(understood(["question"], "Catalog")), Scripted(answer(reply)))
    state = run(graph, "What do you offer?")
    assert state["reply"] == "We offer Plan A and Plan B [KB Section 8: Catalog]. Would you like a quote?"


def test_steps_carry_the_model_and_tokens_of_each_call():
    message = AIMessage(content=CITED, response_metadata={"model_provider": "groq", "model_name": "gpt-oss"},
                        usage_metadata={"input_tokens": 2000, "output_tokens": 300, "total_tokens": 2300})
    graph, _ = make_graph(Scripted(understood(["question"], "Catalog")), Scripted(message))
    state = run(graph, "What do you offer?")
    agent_step = next(s for s in state["steps"] if s["node"] == "agent")
    assert (agent_step["data"]["provider"], agent_step["data"]["model"], agent_step["data"]["tokens"]) == (
        "groq", "gpt-oss", 2300)


def test_previous_turn_messages_are_plain_human_and_ai():
    graph, _ = make_graph(Scripted(understood(["question"], "q")), Scripted(answer("A1"), answer("A2")))
    run(graph, "Q1")
    state = run(graph, "Q2")
    assert isinstance(state["messages"][0], HumanMessage) and state["messages"][0].content == "Q1"

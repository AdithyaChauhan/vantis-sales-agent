"""scripts/seed_demo_leads.py: demo leads through the lead service, the test branch by default, never twice."""
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.db import init_db
from app.leads import service
from app.leads.models import Callback, Lead, SupportRequest
from app.services.service_areas import ServiceAreas
from scripts import seed_demo_leads as demo
from tests.test_answer import FakeLLM

MAIN = "postgresql://u:p@main-host/db?sslmode=require"
TEST = "postgresql://u:p@test-host/db?sslmode=require"
# Synthetic list (not the KB): the demo cities that count as listed; Pune and Mumbai are not on it
AREAS = ServiceAreas.from_rows(
    [{"Region": "North", "Reference Locations": "Noida, Gurugram, Delhi, Jaipur, Kanpur, Lucknow"}]
)
# What the scoring rules make of the demo data (docs/SCORING.md)
EXPECTED = {
    "Priya Verma": (90, "HOT", "listed"),
    "Arjun Mehta": (95, "HOT", "listed"),
    "Karan Malhotra": (87, "HOT", "listed"),
    "Neha Agarwal": (71, "WARM", "listed"),
    "Vikram Singh": (57, "WARM", "listed"),
    "Sana Khan": (74, "WARM", "unverified"),  # 85 before the cap
    "Rohit Patel": (31, "COLD", "unverified"),
    "Ananya Gupta": (41, "COLD", "listed"),
}


def settings(database_url: str, test_database_url: str) -> Settings:
    # _env_file=None: only the values given here, never backend/.env
    return Settings(_env_file=None, database_url=database_url, test_database_url=test_database_url)


def test_writes_to_the_test_branch_by_default_and_refuses_without_one():
    assert demo.target_engine(settings(MAIN, TEST), use_main=False).url.host == "test-host"
    with pytest.raises(SystemExit, match="TEST_DATABASE_URL is not set"):
        demo.target_engine(settings(MAIN, ""), use_main=False)
    with pytest.raises(SystemExit, match="same as DATABASE_URL"):
        demo.target_engine(settings(MAIN, MAIN), use_main=False)


def test_main_needs_the_word_typed_and_never_asks_otherwise():
    asked = []
    for answer in ("", "yes", "Main please"):
        with pytest.raises(SystemExit, match="Not confirmed"):
            demo.target_engine(settings(MAIN, TEST), use_main=True, ask=lambda prompt, a=answer: asked.append(prompt) or a)
    assert demo.target_engine(settings(MAIN, TEST), use_main=True, ask=lambda _: " main ").url.host == "main-host"
    assert len(asked) == 3 and '"main"' in asked[0]
    demo.target_engine(settings(MAIN, TEST), use_main=False, ask=lambda _: pytest.fail("asked for the test branch"))


def test_demo_data_is_clearly_fake_and_marked():
    for raw in demo.DEMO_LEADS:
        assert raw.get("email", "x@example.com").endswith("@example.com")
        assert raw.get("phone", "9000000000").startswith("90000000")
    assert demo.DEMO_SUPPORT["email"].endswith("@example.com")
    assert demo.DEMO_SUPPORT["issue"].startswith("Demo:")
    assert set(demo.CALLBACKS) <= {raw["name"] for raw in demo.DEMO_LEADS}


def test_demo_is_a_lead_channel_and_unknown_ones_still_raise():
    with pytest.raises(ValueError, match="channel"):
        service.create_lead(None, {}, channel="sms")  # checked before any database access


@pytest.fixture
def db(test_engine):
    init_db(test_engine)
    with test_engine.begin() as conn:
        conn.execute(text("TRUNCATE leads, callbacks, support_requests RESTART IDENTITY CASCADE"))
    return test_engine


def counts(db) -> tuple[int, int, int]:
    with Session(db) as session:
        return tuple(session.scalar(select(func.count()).select_from(model)) for model in (Lead, Callback, SupportRequest))


def test_seeds_scored_demo_leads_two_callbacks_and_one_support_request_once(db):
    llm = FakeLLM("A demo reason.")
    first = demo.seed(db, llm=llm, areas=AREAS)
    assert (first.leads_added, first.leads_skipped, first.callbacks_added, first.support_added) == (8, 0, 2, 1)
    assert len(llm.calls) == 8  # one reason per new lead
    assert counts(db) == (8, 2, 1)

    with Session(db) as session:
        leads = session.scalars(select(Lead)).all()
        assert {lead.name: (lead.lead_score, lead.lead_status, lead.location_status) for lead in leads} == EXPECTED
        assert {lead.source_channel for lead in leads} == {"demo"}
        assert {lead.reason_source for lead in leads} == {"llm"}
        by_name = {lead.name: lead for lead in leads}
        sana = by_name["Sana Khan"]
        assert sana.score_breakdown["raw_score"] == 85 and [c["rule"] for c in sana.score_breakdown["caps"]] == [
            "unverified_location"]
        assert sana.recommended_action.endswith("Customer asked for a callback: tomorrow after 5 pm.")
        assert [(c.preferred_time, c.status) for c in by_name["Arjun Mehta"].callbacks] == [("Thursday 11 am", "requested")]
        support = session.scalars(select(SupportRequest)).one()
        assert support.issue.startswith("Demo:") and support.session_id is None

    again = demo.seed(db, llm=llm, areas=AREAS)
    assert (again.leads_added, again.leads_skipped, again.callbacks_added, again.support_added) == (0, 8, 0, 0)
    assert len(llm.calls) == 8  # nothing new, so no new reasons
    assert counts(db) == (8, 2, 1)


def test_a_chat_lead_with_a_demo_name_is_not_mistaken_for_demo_data(db):
    llm = FakeLLM("A demo reason.")
    service.create_lead(db, {**demo.DEMO_LEADS[0]}, llm=llm, areas=AREAS)  # a real "Priya Verma" from the chat
    report = demo.seed(db, llm=llm, areas=AREAS)
    assert report.leads_added == 8
    with Session(db) as session:
        channels = session.scalars(select(Lead.source_channel).where(Lead.name == "Priya Verma")).all()
        assert sorted(channels) == ["demo", "web"]

"""Add demo leads, so the sales dashboard has something to show. Writes to the Neon test branch.

Run: python -m scripts.seed_demo_leads           (the test branch, TEST_DATABASE_URL)
     python -m scripts.seed_demo_leads --reset   (the test branch: empty the lead tables first, then seed)
     python -m scripts.seed_demo_leads --main    (the main database; you must type "main" first)

Every lead goes through app.leads.service, exactly like a chat lead: the same checks, score, status,
caps, recommended action and reason (written by the main model, or the template if that fails; about
one call per new lead). The leads have source_channel "demo", so the dashboard tags them. Contacts are
clearly fake: phone numbers 90000 0000N (valid in format only) and @example.com emails.

--reset (T12a) deletes every lead, callback and support request on the test branch (chat leads from local
checks too) and seeds again, so the dashboard shows the demo data with real reasons. It never runs on main.

Running it again adds nothing: a demo lead with the same name, a callback with the same time for that
lead and a support request with the same name and issue are skipped. The output never shows a phone
number or an email address.
"""
import argparse
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.runnables import Runnable
from sqlalchemy import Engine, create_engine, func, select, text
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.db import init_db
from app.leads import service
from app.leads.models import Callback, Lead, SupportRequest
from app.services.service_areas import ServiceAreas
from scripts.serve_test_branch import engine_for_test_branch

CONFIRM_WORD = "main"

# What each demo customer told us. The service decides the score and status; the comments give what
# the rules (docs/SCORING.md) make of them with the KB Section 10 city list.
DEMO_LEADS = [
    {  # HOT 90: listed city, 5 kW, this month, owner with a rooftop
        "name": "Priya Verma", "phone": "9000000001", "email": "priya.verma@example.com", "city": "Noida",
        "property_type": "house", "ownership": "own", "rooftop_available": True, "system_size_kw": 5,
        "requirement": "5 kW system for our house roof", "monthly_bill_inr": 4500, "timeline": "this month",
        "timeline_bucket": "lt_1m", "intent_level": "interested",
    },
    {  # HOT 95: a 20 kW warehouse roof in 2-3 months; asked for a callback
        "name": "Arjun Mehta", "phone": "9000000002", "email": "arjun.mehta@example.com", "city": "Gurugram",
        "property_type": "warehouse", "business_type": "Logistics", "ownership": "own",
        "rooftop_available": True, "system_size_kw": 20, "requirement": "20 kW on the warehouse roof",
        "timeline": "in two or three months", "timeline_bucket": "1_3m", "intent_level": "ready",
    },
    {  # HOT 87: monthly usage given, no size
        "name": "Karan Malhotra", "phone": "9000000003", "email": "karan.malhotra@example.com",
        "city": "Delhi", "property_type": "clinic", "monthly_units_kwh": 900,
        "requirement": "Solar to run the clinic's air conditioning and equipment", "timeline": "next month",
        "timeline_bucket": "lt_1m", "intent_level": "interested",
    },
    {  # WARM 71: bill given, 4-5 months, phone only
        "name": "Neha Agarwal", "phone": "9000000004", "city": "Jaipur", "property_type": "house",
        "monthly_bill_inr": 3500, "requirement": "Want to lower my electricity bill",
        "timeline": "in four or five months", "timeline_bucket": "3_6m", "intent_level": "interested",
    },
    {  # WARM 57: the need only in words, no timeline
        "name": "Vikram Singh", "phone": "9000000005", "email": "vikram.singh@example.com", "city": "Kanpur",
        "property_type": "shop", "business_type": "Hardware shop",
        "requirement": "Solar on the roof of my shop to cut the power bill", "intent_level": "interested",
    },
    {  # WARM 74: would be 85, but Pune is not in the listed service areas (capped); asked for a callback
        "name": "Sana Khan", "phone": "9000000006", "email": "sana.khan@example.com", "city": "Pune",
        "property_type": "apartment", "ownership": "own", "rooftop_available": True, "system_size_kw": 4,
        "requirement": "4 kW for the terrace of our flat", "timeline": "as soon as possible",
        "timeline_bucket": "lt_1m", "intent_level": "ready",
    },
    {  # COLD 31: unlisted city, only exploring, free text as customers really type it
        "name": "Rohit Patel", "email": "rohit.patel@example.com", "city": "Mumbai", "property_type": "office",
        "requirement": "i want to install one at my organisation", "timeline": "maybe next year",
        "timeline_bucket": "gt_6m", "intent_level": "exploring",
        "other_info": "Asked for the whole office building, not just his floor",
    },
    {  # COLD 41: renting, just looking
        "name": "Ananya Gupta", "phone": "9000000008", "city": "Lucknow", "property_type": "apartment",
        "ownership": "rent", "requirement": "Just checking what solar would involve, for later",
        "intent_level": "exploring",
    },
]
CALLBACKS = {"Arjun Mehta": "Thursday 11 am", "Sana Khan": "tomorrow after 5 pm"}
# No source_channel column on support requests (Adithya's choice): "Demo:" is for people reading the
# list only; no code looks for it.
DEMO_SUPPORT = {
    "name": "Meera Iyer", "phone": "9000000009", "email": "meera.iyer@example.com",
    "issue": "Demo: the inverter shows a red error light since Monday",
    "project_details": "3 kW rooftop system installed in 2025",
}


@dataclass
class SeedReport:
    """What one run added (and how many demo leads were already there)."""

    leads_added: int = 0
    leads_skipped: int = 0
    callbacks_added: int = 0
    support_added: int = 0


def _demo_lead_id(engine: Engine, name: str) -> int | None:
    with Session(engine) as session:
        return session.scalar(select(Lead.id).where(Lead.source_channel == "demo", Lead.name == name))


def _has_callback(engine: Engine, lead_id: int, preferred_time: str) -> bool:
    with Session(engine) as session:
        found = select(Callback.id).where(Callback.lead_id == lead_id, Callback.preferred_time == preferred_time)
        return session.scalar(found) is not None


def _has_support_request(engine: Engine, name: str, issue: str) -> bool:
    with Session(engine) as session:
        found = select(SupportRequest.id).where(SupportRequest.name == name, SupportRequest.issue == issue)
        return session.scalar(found) is not None


def seed(engine: Engine, *, llm: Runnable | None = None, areas: ServiceAreas | None = None) -> SeedReport:
    """Add whatever demo data is not there yet. llm and areas default to the service's (tests pass fakes)."""
    report = SeedReport()
    for raw in DEMO_LEADS:
        lead_id = _demo_lead_id(engine, raw["name"])
        if lead_id is None:
            result = service.create_lead(engine, raw, channel="demo", llm=llm, areas=areas)
            if result.lead is None or result.errors:  # the demo data itself is wrong: stop loudly
                raise RuntimeError(f"{raw['name']}: missing {result.missing_fields}, invalid {sorted(result.errors)}")
            lead_id = result.lead.id
            report.leads_added += 1
        else:
            report.leads_skipped += 1
        preferred_time = CALLBACKS.get(raw["name"])
        if preferred_time and not _has_callback(engine, lead_id, preferred_time):
            service.schedule_callback(engine, lead_id, preferred_time)
            report.callbacks_added += 1
    if not _has_support_request(engine, DEMO_SUPPORT["name"], DEMO_SUPPORT["issue"]):
        result = service.create_support_request(engine, DEMO_SUPPORT)
        if result.id is None or result.errors:
            raise RuntimeError(f"support request: missing {result.missing_fields}, invalid {sorted(result.errors)}")
        report.support_added += 1
    return report


def reset(engine: Engine) -> tuple[int, int, int]:
    """Delete every lead, callback and support request (ids start again at 1). Returns how many there were.
    main() only calls it for the test branch."""
    with Session(engine) as session:
        counts = tuple(int(session.scalar(select(func.count()).select_from(model)))
                       for model in (Lead, Callback, SupportRequest))
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE leads, callbacks, support_requests RESTART IDENTITY CASCADE"))
    return counts


def target_engine(settings: Settings, use_main: bool, ask: Callable[[str], str] = input) -> Engine:
    """The test branch (refuses an empty or same-as-main TEST_DATABASE_URL), or main after "main" is typed.

    Does not connect.
    """
    if not use_main:
        return engine_for_test_branch(settings)
    if not settings.sqlalchemy_database_url:
        raise SystemExit("DATABASE_URL is not set; nothing to write to.")
    print(f"This adds up to {len(DEMO_LEADS)} demo leads, {len(CALLBACKS)} callbacks and 1 support request "
          "to the MAIN database (DATABASE_URL).")
    if ask(f'Type "{CONFIRM_WORD}" to continue: ').strip() != CONFIRM_WORD:
        raise SystemExit("Not confirmed; nothing was written.")
    return create_engine(settings.sqlalchemy_database_url, pool_pre_ping=True, connect_args={"connect_timeout": 10})


def print_summary(engine: Engine, report: SeedReport, where: str) -> None:
    """The demo leads as the service scored them (no phone numbers or emails)."""
    print(f"\nOn {where}: added {report.leads_added} demo leads ({report.leads_skipped} were already there), "
          f"{report.callbacks_added} callbacks, {report.support_added} support request.")
    with Session(engine) as session:
        leads = session.scalars(select(Lead).where(Lead.source_channel == "demo").order_by(Lead.id)).all()
        print(f"{'id':>4}  {'name':<16}{'city':<10}{'location':<12}{'score':>5}  {'status':<7}{'source':<9}"
              f"{'callback':<22}reason")
        for lead in leads:
            callback = lead.callbacks[-1].preferred_time if lead.callbacks else "-"
            print(f"{lead.id:>4}  {lead.name:<16}{lead.city:<10}{lead.location_status:<12}{lead.lead_score:>5}  "
                  f"{lead.lead_status:<7}{lead.reason_source:<9}{callback:<22}{lead.reason}")


def main(argv: list[str] | None = None) -> None:
    """Parse --main and --reset, pick the database, create missing tables, (empty them,) seed and print."""
    parser = argparse.ArgumentParser(description="Add demo leads to the test branch (or to main with --main).")
    parser.add_argument("--main", action="store_true", help='write to the main database (asks you to type "main")')
    parser.add_argument("--reset", action="store_true",
                        help="test branch only: delete every lead, callback and support request first")
    args = parser.parse_args(argv)
    if args.reset and args.main:
        raise SystemExit("--reset works on the test branch only; it can't be combined with --main.")
    engine = target_engine(get_settings(), args.main)
    init_db(engine)
    if args.reset:
        leads, callbacks, support = reset(engine)
        print(f"Reset the test branch: deleted {leads} leads, {callbacks} callbacks and {support} support requests.")
    report = seed(engine)
    print_summary(engine, report, "main" if args.main else "the test branch")


if __name__ == "__main__":
    main()

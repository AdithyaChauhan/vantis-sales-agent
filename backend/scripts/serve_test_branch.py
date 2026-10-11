"""Run the API locally with leads, callbacks and support requests on the Neon test branch.

Dev only, never for production. The KB and the service areas are still read from DATABASE_URL (main),
but the chat agent writes leads to TEST_DATABASE_URL and the sales API reads and writes them there too;
chat memory stays in this process (gone on restart). Use it for local checks of the chat page and the
dashboard, so they never put leads into main.

Run: python -m scripts.serve_test_branch   (then the chat page: cd frontend && npm run dev)
"""
import os
from contextlib import asynccontextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import Settings, to_sqlalchemy_url

PORT = 8000
# Set before the app is imported (its settings are read once). Values already in the environment win.
DEV_ENV = {"CORS_ORIGINS": "http://localhost:5173", "CHAT_SHOW_LEAD_SCORE": "true"}


def engine_for_test_branch(settings: Settings) -> Engine:
    """Engine for TEST_DATABASE_URL. Refuses (SystemExit) when it is empty or the same as DATABASE_URL,
    so this script can never write leads to the main database. Does not connect."""
    if not settings.test_database_url:
        raise SystemExit("TEST_DATABASE_URL is not set in backend/.env; refusing to start.")
    url = to_sqlalchemy_url(settings.test_database_url)
    if url == settings.sqlalchemy_database_url:
        raise SystemExit("TEST_DATABASE_URL is the same as DATABASE_URL; refusing to start.")
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})


def main() -> None:
    """Start uvicorn on 127.0.0.1:8000 with the agent's lead writes pointed at the test branch."""
    for key, value in DEV_ENV.items():
        os.environ.setdefault(key, value)

    import uvicorn
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import build_agent, get_agent
    from app.api.sales import leads_engine
    from app.core.config import get_settings
    from app.core.db import get_engine, init_db
    from app.main import app

    test_engine = engine_for_test_branch(get_settings())

    @asynccontextmanager
    async def lifespan(_app):
        init_db(test_engine)  # lead tables on the test branch; main is only read
        yield

    app.router.lifespan_context = lifespan
    agent = build_agent(get_engine(), InMemorySaver(), leads_engine=test_engine)
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[leads_engine] = lambda: test_engine  # the sales API (dashboard)
    print("Leads, callbacks and support requests are on the test branch.", flush=True)
    print(f"API on http://127.0.0.1:{PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()

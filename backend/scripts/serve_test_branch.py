"""Run the API locally with leads, callbacks and support requests on the Neon test branch.

Dev only, never for production. The KB and the service areas are still read from DATABASE_URL (main),
but the chat agent writes leads to TEST_DATABASE_URL and the sales API reads and writes them there too;
chat memory stays in this process (gone on restart). Use it for local checks of the chat page and the
dashboard, so they never put leads into main.

Run: python -m scripts.serve_test_branch                 (then the chat page: cd frontend && npm run dev)
     python -m scripts.serve_test_branch --slow-gemini   (the slow-Google drill: every Gemini chat call hangs)
     python -m scripts.serve_test_branch --port 8001     (another port, e.g. when an older server holds 8000)

--slow-gemini (T12a) exists only here: it is a command-line flag, not a setting, and the Docker image does not
contain backend/scripts, so it can never be switched on in Render.
"""
import argparse
import os
import time
from contextlib import asynccontextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core import llm
from app.core.config import Settings, to_sqlalchemy_url

PORT = 8000
# Set before the app is imported (its settings are read once). Values already in the environment win.
DEV_ENV = {"CORS_ORIGINS": "http://localhost:5173", "CHAT_SHOW_LEAD_SCORE": "true"}
HANG_S = llm.GEMINI_MIN_TIMEOUT_S  # how long a forced call hangs: like a real Gemini call that hits its HTTP timeout


class HangingGemini(llm.GeminiChat):
    """--slow-gemini: every call hangs like an overloaded Google endpoint, then times out. The app's own
    deadlines cut it off first, so the drill exercises the real fallbacks and the skip list."""

    def _generate(self, *args, **kwargs):
        time.sleep(HANG_S)
        raise TimeoutError("forced Gemini timeout (serve_test_branch --slow-gemini)")


def force_gemini_timeouts() -> None:
    """Make every Gemini chat model built from now on hang (the embedder is not touched). Call it before the
    agent is built."""

    def hanging_gemini_chat(model: str, settings: Settings) -> llm.GeminiChat:
        return HangingGemini(model=model, google_api_key=settings.google_api_key or "unused",
                             thinking_level=llm.THINKING_LEVEL)

    llm.gemini_chat = hanging_gemini_chat
    llm.reason_llm.cache_clear()  # in case it was built already


def engine_for_test_branch(settings: Settings) -> Engine:
    """Engine for TEST_DATABASE_URL. Refuses (SystemExit) when it is empty or the same as DATABASE_URL,
    so this script can never write leads to the main database. Does not connect."""
    if not settings.test_database_url:
        raise SystemExit("TEST_DATABASE_URL is not set in backend/.env; refusing to start.")
    url = to_sqlalchemy_url(settings.test_database_url)
    if url == settings.sqlalchemy_database_url:
        raise SystemExit("TEST_DATABASE_URL is the same as DATABASE_URL; refusing to start.")
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """--slow-gemini and --port (T12b: another port when an older server holds 8000)."""
    parser = argparse.ArgumentParser(description="Run the API locally with leads on the Neon test branch.")
    parser.add_argument("--slow-gemini", action="store_true",
                        help="drill: every Gemini chat call hangs, so the deadlines, Groq and the skip list answer")
    parser.add_argument("--port", type=int, default=PORT, help=f"port to listen on (default {PORT})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Start uvicorn on 127.0.0.1 (port 8000 by default) with the agent's lead writes pointed at the test branch."""
    args = parse_args(argv)
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
    if args.slow_gemini:
        force_gemini_timeouts()

    @asynccontextmanager
    async def lifespan(_app):
        init_db(test_engine)  # lead tables on the test branch; main is only read
        yield

    app.router.lifespan_context = lifespan
    agent = build_agent(get_engine(), InMemorySaver(), leads_engine=test_engine)
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[leads_engine] = lambda: test_engine  # the sales API (dashboard)
    print("Leads, callbacks and support requests are on the test branch.", flush=True)
    if args.slow_gemini:
        print(f"Drill: every Gemini chat call hangs for {HANG_S} s (--slow-gemini).", flush=True)
    print(f"API on http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

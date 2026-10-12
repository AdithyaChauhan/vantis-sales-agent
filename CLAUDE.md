# CLAUDE.md: Vantis Sales Agent

Read this at the start of every session.
- Full spec, decisions and task list: `docs/PLAN.md`
- Where we are: `docs/PROGRESS.md` (newest entry at the bottom)
- How the code is built (once code exists): `docs/ARCHITECTURE.md`

## Project
Pradrix AI Engineer Intern, Round 2 assignment. Deadline: Thu 15 Oct 2026, 23:59 IST (target: submit by 20:00).
Build an AI sales and lead-qualification agent for **Vantis Energy Solutions** (fictional solar company).
Graded on: agent workflow 20%, RAG 20%, tool calling 15%, prompts 10%, API 10%, lead scoring 10%, code 5%, docs 5%, demo 5%.
Goal: production-minded, explainable and tested. Not flashy.
The owner (Adithya) must be able to explain every line in an interview. Prefer clear code over clever code.

## Stack (decided; ask before changing anything here)
- Backend: Python 3.11 or 3.12, FastAPI, Pydantic v2, SQLAlchemy 2 + psycopg 3
- Agent: LangGraph + langchain-core. Chat memory: LangGraph Postgres checkpointer
- LLM: Gemini (main model for the agent, fast model for small steps), with automatic fallback to Groq. Model names come from env, never hardcoded
- Embeddings: Gemini embedding model (from env), stored in pgvector
- Database: Neon Postgres with pgvector. One DB for KB chunks, service areas, leads, callbacks, support requests and chat memory
- Frontend: React + Vite + plain CSS
- Deploy: backend to Render (Dockerfile), frontend to Vercel, DB on Neon
- Tests: pytest. The LLM is always faked in unit and API tests

## Layout
```
backend/app/        FastAPI app: api/, agent/, rag/, leads/, services/, core/
backend/tests/      pytest
backend/scripts/    one-off scripts (list_models, eval_retrieval)
frontend/           React app (chat page + sales dashboard)
prompts/            every LLM prompt as .md, plus prompts/README.md (log of changes)
data/kb/            the Vantis KB PDF (local only, gitignored)
data/eval/          retrieval test questions
docs/               PLAN.md, PROGRESS.md, ARCHITECTURE.md, EVAL.md
```

## Commands (Windows + Git Bash)
- venv: `cd backend && source .venv/Scripts/activate`
- API: `cd backend && uvicorn app.main:app --reload --port 8000`
- Tests: `cd backend && python -m pytest -q` (real-LLM tests: `python -m pytest -m live`)
- Voice check, Adithya's 4-turn session with replies and step times (about 11 LLM calls, paced; prints "→", so set UTF-8): `cd backend && PYTHONIOENCODING=utf-8 python -m pytest -m live -s tests/test_live_voice.py`
- List Gemini models: `cd backend && python -m scripts.list_models`
- Ingest KB dry run (parse and print chunks, no DB): `cd backend && python -m app.rag.ingest --dry-run`
- Ingest KB (embeds, then replaces the KB rows in one transaction): `cd backend && python -m app.rag.ingest`
- Search the KB: `cd backend && python -m scripts.search_kb "question" [--mode keyword|vector|hybrid]`
- Retrieval eval: `cd backend && python -m scripts.eval_retrieval`
- Retrieval eval with the agent's query rewrite (about 32 fast-LLM calls, paced): `cd backend && python -m scripts.eval_retrieval --rewrite`
- API for local page checks, leads on the Neon test branch (never main): `cd backend && python -m scripts.serve_test_branch` (for the dashboard add a throwaway key: `SALES_API_KEY=any-local-key-of-16-chars python -m scripts.serve_test_branch`)
- Demo leads on the test branch (8 leads, 2 callbacks, 1 support request; about 8 fast-LLM calls for the reasons; a second run adds nothing; pytest empties these tables, so re-run it after tests): `cd backend && python -m scripts.seed_demo_leads` (`--main` writes to main only after you type "main")
- Frontend setup: `cd frontend && npm install`
- Frontend: `cd frontend && npm run dev` (http://localhost:5173; calls the API at `VITE_API_URL`, default http://127.0.0.1:8000; the sales dashboard is at /leads and /support and asks for the sales API key)
- Frontend tests: `cd frontend && npm test`
- Frontend build: `cd frontend && npm run build`
- Docker image (repo root, Docker Desktop running): `docker build -t vantis-api .`
- Run the image: `docker run --rm -p 10000:10000 --env-file backend/.env vantis-api` (add `--cpus=0.1 --memory=512m` to match Render's free plan)
- Deploy: push to `main`; Render builds from `render.yaml` (steps in `docs/DEPLOY.md`)
These commands appear as their tasks get built. Keep this list in sync.

## Rules
1. Work only on the task you were given. For anything touching more than 2 files, show a plan first and wait for approval.
2. "Done" means you ran the tests and they passed. Never say something works without running it.
3. Business facts (products, cities, warranty, pricing rules) come only from the KB through ingestion. Never hardcode them in code or prompts. Exception: the small city-alias map (e.g. Gurgaon → Gurugram).
4. All prompts live in `/prompts` as `.md` files and code loads them. Every prompt change gets a line in `prompts/README.md` (what changed, why, the example that failed before).
5. Secrets come only from env vars. Never read, print or commit `.env` files. Keep `backend/.env.example` in sync with the code.
6. Never commit `.env` files, the assignment PDF, or the KB PDF in `data/kb/`. The KB PDF stays local; the README tells reviewers where to put their copy.
7. Keep it simple. No new libraries, services or abstraction layers unless the task needs them; say why when you add one.
8. Type hints and short docstrings on public functions. Small modules, plain names.
9. The chat must never crash. Catch errors, log them with the request id, and return a friendly message.
10. If something still fails after 2 real attempts, stop and explain what you tried instead of guessing.
11. Finish every task with `/handoff`.

## Git
- Commit only through `/handoff`, after tests pass. Use conventional messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Never push, force-push or rewrite history. Adithya pushes.

## Gotchas
- Windows + Git Bash: use forward slashes. The venv Python is `.venv/Scripts/python`.
- Neon: use the direct (non-pooled) connection string and enable pre-ping. The DB sleeps after 5 idle minutes.
- Free LLM tiers allow only a few requests per minute. Use retries with backoff, never call the real LLM in unit tests, and keep prompts short.
- `.gitattributes` forces LF line endings. Render builds Linux containers.

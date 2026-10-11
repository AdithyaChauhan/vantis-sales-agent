# PROGRESS.md

Newest entry at the bottom. Claude Code appends one entry per task through `/handoff`. Keep each entry to 12 lines or fewer.
Adithya pastes the newest entry to the reviewer (Claude chat) after every task.

## Entry template
```
## T<n>: <title> (<YYYY-MM-DD>)
- Status: done | partial | blocked
- Changed: <main files or modules, one line>
- Tests: `<command>` → <N passed, M failed, K skipped>
- Also ran: <what you actually ran besides tests, e.g. GET /api/health → 200 db ok>
- Prompt changes: none | <file: what and why>
- Decisions: none | <short>
- Open issues: none | <short>
- Next: <next task>
```

---

## T0: Plan and setup (2026-10-10)
- Status: done
- Changed: CLAUDE.md, docs/PLAN.md, docs/PROGRESS.md, .claude/settings.json, .claude/skills/handoff/SKILL.md, data/eval/retrieval_eval.jsonl, .gitignore, .gitattributes
- Tests: none (no code yet)
- Decisions: stack per PLAN.md part 3; Gemini main + Groq fallback; Neon; Render + Vercel; mock WhatsApp skipped; KB PDF stays out of git (README says where to put it)
- Open issues: check the submission form fields · check free-tier limits in the AI Studio and Groq consoles
- Next: T1 Foundation

## T1: Foundation (2026-10-10)
- Status: done
- Changed: backend/ skeleton (app/main.py, core/config|logging|db, api/health, scripts/list_models.py, tests/), requirements.txt, .env.example, .gitkeep files, PLAN.md KB path, CLAUDE.md commands
- Tests: `cd backend && python -m pytest -q` → 5 passed, 0 failed, 0 skipped
- Also ran: uvicorn + GET /api/health → {"status":"ok","db":"ok"}, startup log "pgvector extension ready"; `python -m scripts.list_models` → 45 chat, 3 embedding models
- Prompt changes: none
- Decisions: Python 3.12 venv; KB_PATH default is the original PDF name Vantis_Energy_Solutions_Knowledge_Base.pdf; stdlib JSON logging (no extra lib); health returns 200 "degraded" when DB is down
- Open issues: pick LLM_MODEL_MAIN / LLM_MODEL_FAST / EMBEDDING_MODEL from list_models and check free-tier limits; EMBEDDING_DIM must match T3 vector column
- Next: T2 Ingestion part 1

## T2: Ingestion part 1 (2026-10-10)
- Status: done
- Changed: app/rag/ (pdf_parser, chunker, ingest), app/services/service_areas.py, 3 test files, requirements.txt (pdfplumber 0.11.10, rapidfuzz 3.14.6), docs (no section sign; citation format [KB Section N: Title]), CLAUDE.md commands
- Tests: `cd backend && python -m pytest -q` → 37 passed, 0 failed, 0 skipped (with the PDF absent: 14 passed, 19 skipped)
- Also ran: `python -m app.rag.ingest --dry-run` → 19 sections, 52 chunks (prose 11, table_row 26, faq 11, policy 4); KB Section 8: 7 rows + 1 prose, 9: 5, 10: 5 + 1 prose, 11: 5, 12: 4, 15: 11 faq, 16-19: 1 policy each, 1-7 and 13-14: 1 prose each; missing PDF → clear message, exit 1
- Prompt changes: none
- Decisions: PDF quirks fixed in the parser (bullet shown as "(cid:127)", arrow extracted as "fi"); rapidfuzz `ratio` scorer (not WRatio) to avoid partial-match false positives; alias target must be in the KB list; state only narrows regions
- Open issues: none (service-area lookup reads the PDF for now; T3 moves it to the DB table)
- Next: T3 Embeddings, tables, hybrid retriever


## T3: Embeddings, tables, full ingest, hybrid retriever (2026-10-11)
- Status: done
- Changed: app/core/llm.py (embedder), app/rag/models.py (kb_chunks, service_areas), retriever.py, ingest.py (full run), service_areas.py (reads the table), health (kb_chunks), scripts/search_kb.py, 4 new test files, pgvector 0.5.1 pinned (Vector column type, binds lists), PLAN.md 3.6/3.7, .env.example
- Tests: `cd backend && python -m pytest -q` → 77 passed, 0 failed; `python -m pytest -m live` → 1 passed
- Also ran: probe (2 texts → 2 vectors of 768); `python -m app.rag.ingest` once → 52 chunks, 19 service-area rows, doc_version fb9fc76a08fe; `search_kb "Is there a guarantee on the panels?"` → KB Section 11 is #1; "Do you provide installation in Delhi?" → KB Section 10 is only #4 (Sections 6, 1, 15 above it); GET /api/health → kb_chunks 52; --dry-run still 52 chunks with no DB or key
- Prompt changes: none
- Decisions: models gemini-3.5-flash-lite (main), gemini-3.1-flash-lite (fast), gemini-embedding-2 at 768 dims, openai/gpt-oss-120b fallback (from T5). Free tier: Gemini Flash only 20 req/day; the two Flash-Lite models 15/min and 500/day each (separate quotas); embedding 100/min, 30K tokens/min, 1,000/day; Groq 30/min, 1,000/day, 8K tokens/min. One Content per text (else one merged vector). Query embedding 10 s + 1 retry, ingest 30 s + 2 retries. All DB write functions take the engine as an argument; test_engine skips if it equals DATABASE_URL. Gate threshold 0.63 (KB questions 0.73-0.78, off-topic 0.51-0.54, off-topic had 0 keyword hits)
- Open issues: OR'd keyword search lets generic words ("provide", "installation") match 10 chunks and push Section 10 to #4 for the Delhi question; T4 to tune. Retriever returns chunks even when no_context is true, so T5 must check the flag. City typo "Dehli" returns unverified (rapidfuzz ratio 80, below 85); fix in T12 by allowing one swapped pair of letters or adding common misspellings to the alias list
- Next: T4 Retrieval eval script + docs/EVAL.md

## T4: Retrieval eval + tuning (2026-10-11)
- Status: done
- Changed: scripts/eval_retrieval.py, data/eval/offtopic_eval.jsonl and docs/EVAL.md (new); app/rag/retriever.py (common-word filter, keyword weight, gate rule); tests (new test_eval_retrieval.py, updated test_retriever.py and test_kb_db.py); PLAN.md 3.6, ARCHITECTURE.md, config and .env.example comments
- Tests: `cd backend && python -m pytest -q` → 102 passed, 0 failed, 0 skipped (1 live deselected); `python -m pytest -m live` → 1 passed
- Also ran: `python -m scripts.eval_retrieval` once (32 embedding calls) plus one tuning run with a scratchpad script (32 calls), so 2 of the 4 allowed runs → hybrid hit@1 21/24, hit@3 22/24 (0.917, was 20/24), MRR 0.907 (was 0.807); vector hit@3 22/24; keyword hit@3 20/24 (was 18). `search_kb "Do you provide installation in Delhi?"` → KB Section 10 is #1 (was #4). Gate: hybrid blocks 8/8 off-topic and 0/24 KB questions
- Prompt changes: none
- Decisions: kept the common-word filter (drop words in more than 30% of chunks; 25-35% identical, 40% loses the gain) and keyword weight 0.5 in RRF (only MRR +0.006, one constant); removed shared tie ranks (no gain on top of the filter). Gate: with a vector search only similarity counts, since 3 of 8 off-topic questions had keyword hits; keyword hits count only in the keyword-only fallback. Threshold stays 0.63 (KB 0.676-0.837, off-topic 0.492-0.576). Eval labels untouched
- Open issues: hybrid misses q01 "What solar systems do you provide?" (catalog rows at #6 and #7, so T5 needs a sharper query or a search_knowledge_base call) and q09 Pune (check_service_area covers it in the agent); keyword-only fallback blocks only 5/8 off-topic and finds nothing for Gurgaon; word stats are cached per process (restart after re-ingest); tuned and reported on the same 24 questions; city typo "Dehli" returns unverified (rapidfuzz ratio 80, below 85), fix in T12 by allowing one swapped pair of letters or adding common misspellings to the alias list
- Next: T5 LLM module (Gemini + Groq fallback), rag_prompt.md, grounded /api/chat v0

## T5: LLM layer, prompts, grounded /api/chat v0 (2026-10-11)
- Status: done (q01 moved to T8, by decision)
- Changed: app/core/llm.py (main_llm, fast_llm, chat_llm, invoke_llm, query_embedder), app/core/prompts.py, app/rag/answer.py, app/api/chat.py (POST /api/chat v0), chunker (table_summary for KB Sections 8 and 10), ingest, prompts/ (system_prompt, rag_prompt, README), 4 new test files, requirements (langchain-core 1.6.9, langchain-google-genai 4.4.1, langchain-groq 1.1.3; google-genai stays 2.29.0), PLAN (3.6, 3.7, F0 welcome, T8 notes), EVAL.md T5 note, handoff skill step 8
- Tests: `cd backend && python -m pytest -q` → 144 passed, 0 failed (7 live deselected); `python -m pytest -m live` → 6 passed, 1 xfailed (q01, strict)
- Also ran: dry run → 54 chunks; ingest once; before/after rank snapshot with saved vectors (no question worse; q07 keyword 2 → 1); `eval_retrieval` once → hybrid hit@1 21/24, hit@3 22/24, MRR 0.907 (unchanged), gate 8/8 and 0/24; one real call (thinking minimal) → gemini 1,361 ms; uvicorn + POST /api/chat for the 5 PDF questions → all 200, gemini 0.9-2.0 s, KB Sections 1+15, 5, 10, 6, 15+11 cited; forced fallback → groq 1.2 s
- Prompt changes: system_prompt.md and rag_prompt.md v0; system_prompt.md: start with the answer, no greeting unless the customer greets (warranty reply opened with "Hello!")
- Decisions: Gemini 15 s, 1 retry on timeout/408/5xx, none on 429 (straight to Groq); Groq 15 s, no retry, reasoning_effort low; worst case about 46 s. Reply = text blocks only, `<think>` stripped, Groq's 【 】 turned into [ ]. Sources: one per cited section (all context sections if none cited). No context → fixed reply, no LLM call. DB down → 503; LLM down → 200 degraded
- Open issues: q01 misses KB Sections 8/9 (no keyword hit, summary chunk vector rank 6) → T8 search_query rewrite + small-to-big, see PLAN "T8 notes"; the 5 kW cost reply doesn't mention HomePro 5 ("5kW" vs "5 kW"?) → T8 or T12; T6 Docker image must include /prompts; still open from T4: q09 Pune (check_service_area in T8), keyword-only fallback blocks only 5/8 off-topic and finds nothing for Gurgaon, word stats cached per process (restart API after re-ingest), tuned and reported on the same 24 questions, "Dehli" typo → unverified (T12)
- Next: T6 Dockerfile + deploy the skeleton to Render

## T6: Docker image + first deploy to Render (2026-10-11)
- Status: done (Phase 1 local checks, Phase 2 live checks)
- Changed: Dockerfile, .dockerignore, render.yaml (free, singapore, healthCheckPath /api/health/live, buildFilter), GET /api/health/live (no DB, not logged), main.py quiet paths, docs/DEPLOY.md, tests (test_deploy_config.py, 2 health tests), requirements (PyYAML 6.0.3 pinned, already installed via langchain-core), .env.example comment, CLAUDE.md commands, PLAN (3.9, part 6, T6 result, T9/T13/T14 notes), ARCHITECTURE (deployment diagram)
- Tests: `cd backend && python -m pytest -q` → 151 passed, 0 failed (7 live deselected)
- Also ran: local `docker build` → 72 s, 508 MB, root scan: no .env or PDF, user appuser; `docker run --cpus=0.1 --memory=512m` → start 43 s (7 s at full CPU), 102 → 114 MiB after a chat, chat 6.3 s. Live on Render (vantis-sales-agent-api, Singapore): /api/health → ok, db ok, kb_chunks 54 in 0.13-0.23 s; /api/chat Delhi → 200, not degraded, KB Section 10, gemini, 1.03 s and 1.08 s (repeat); Kota → 1.82 s; after 17 idle minutes: cold /api/health 32.5 s, then chat 3.0 s
- Prompt changes: none
- Decisions: Render checks /api/health/live (every few seconds, 5 s timeout, restart after 60 s; the full check would keep Neon awake and could restart a healthy app during a slow Neon wake); SALES_API_KEY generateValue; secrets sync: false; buildFilter (500 build minutes a month); startup cost is Python imports, left as is (live cold start 32.5 s is acceptable); Render's health checks do not keep the free service awake
- Open issues: buildFilter not yet seen working (expect no deploy in Render Events for this docs-only push); URL stays private until T9 rate limit; T9: real client IP behind Render's proxy; T13: autoDeployTrigger → checksPass, Vercel URL in CORS_ORIGINS; still open from T5: q01 (T8), 5 kW reply without HomePro 5 (T8/T12); from T4: q09 Pune, keyword-only fallback gaps, word stats cached per process, same 24 questions for tuning and reporting, "Dehli" typo (T12)
- Next: T7 Leads layer (tables, validation, scoring, reason, unit tests)

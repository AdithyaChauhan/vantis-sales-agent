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

## T7: Leads layer (2026-10-11)
- Status: done
- Changed: app/leads/ (schemas: LeadData contract + LeadView; validators; scoring + actions; reason; models: leads, callbacks, support_requests; service: create/update/get/list leads, schedule_callback, create_support_request), init_db registers the lead tables, prompts/lead_qualification_prompt.md, docs/SCORING.md, PLAN 3.3/3.4/3.10 + T8 notes, 5 test files + tests/leads_data.py, requirements (email-validator 2.3.0)
- Tests: `cd backend && python -m pytest -q` → 281 passed, 0 failed (8 live deselected); `python -m pytest -m live -k leads` → 1 passed
- Also ran: test-branch round trip (Rahul in Pune 74 WARM → bill change keeps the reason, no LLM call → city Lucknow 97 HOT, new reason → get/list); lead tests make 0 external HTTP calls; real fast_llm reasons: Rahul in Lucknow 97 HOT "…ready to buy a 5 kW system…", Rahul in Pune raw 82 → 74 WARM "…capped due to city not in listed service areas."
- Prompt changes: lead_qualification_prompt.md v0; then "a capped lead's reason names the cap, in the cap's words" (Pune reason first gave no cause, then said "outside service areas")
- Decisions: all 13 plan recommendations accepted: caps limit the score to 74 too; clear requirement = size, bill or usage (no package names in code); null = unknown; property "partly known" = 2; HOT action "Contact the lead within 1 hour and offer a site assessment."; one-sentence reason with a template fallback (failure, empty, long, two sentences, other status, KB Section 16 topics); no personal details or money amounts sent to the LLM; reason regenerated only when the breakdown changes and written outside the DB transaction; output names lead_score / lead_status / score_breakdown / reason / recommended_action; email-validator added (syntax only, no network)
- Open issues: reason latency inside create_lead/update_lead (up to about 46 s worst case) → T8 may move it to finalize; package name → kW mapping is T8's understand step; unique session_id race not handled (one turn at a time per session); push creates the 3 tables in the main DB on the next Render start; still open from T6: buildFilter not yet seen working, URL private until T9 rate limit, client IP behind Render's proxy (T9), checksPass + CORS (T13); from T5: q01 (T8), 5 kW reply without HomePro 5 (T8/T12); from T4: q09 Pune, keyword-only fallback gaps, word stats cached per process, same 24 questions for tuning and reporting, "Dehli" typo (T12)
- Next: T8 LangGraph agent, the 6 tools, checkpointer, extraction prompt, /api/chat v1

## T8a: LangGraph agent part 1 (2026-10-11)
- Status: done
- Changed: app/agent/ (state, understand, tools, graph, memory, welcome), /api/chat through the graph, GET /api/welcome, llm.py (GeminiChat, chat_llm tools/schema, provider_of), answer.py reduced to helpers (v0 answer() removed), eval_retrieval --rewrite, prompts (lead_extraction new; system, rag agent versions), 6 new test files (incl. test_prompt_examples), requirements (langgraph 1.2.14, langgraph-checkpoint-postgres 3.1.2, psycopg-pool 3.3.3; langchain/google-genai pins unchanged), PLAN (T8a/T8b split, 3.1-3.9, T8b/T13 notes, known limitations), ARCHITECTURE, EVAL, CLAUDE.md
- Tests: `cd backend && python -m pytest -q` → 349 passed, 0 failed (10 live deselected); `pytest -m live tests/test_live_chat.py` → 8 passed (24 LLM calls), Kanpur rerun 1 passed (6), test_live_leads + test_live_retrieval → 2 passed
- Also ran: `eval_retrieval --rewrite` twice (32 calls each): rewrite hit@3 24/24 both runs (hit@1 23 → 24 with the new examples), q01 6 → 1, q09 10 → 1, off-topic 8/8 blocked; uvicorn on main: q01 names all 5 packages (KB Section 8), Delhi → "And in Kanpur?" answered from Postgres memory, Pune "not in the listed service areas; the team can check", "hi" → welcome; warm turn about 6 s (understand 1.2-1.7 s, search about 1.1 s, agent 1.8-2.3 s, about 1.3 s checkpoint + HTTP), first request of a process 20 s; Docker at 0.1 CPU/512 MB: 116 MiB after startup, 133 MiB after a turn, startup 93 s (was 43 s; imports, noisy locally)
- Prompt changes: lead_extraction_prompt.md v0, then "rewrite in the KB's words, don't copy" (q01 query copied verbatim), then examples replaced (old ones copied/paraphrased eval and live questions); system_prompt.md agent version (tools, city wording, list every offering, short replies, no claimed actions); rag_prompt.md for multi-turn
- Decisions: GeminiChat adds http_options in _prepare_request (with_structured_output rejects them; private method, pinned + tested); Groq parallel_tool_calls=False; rewrite alone (fused not needed); small-to-big off (fires on 5 unrelated questions); greeting/thanks/off-topic short-circuits only when that intent is alone; "yes"/"no thanks" always to the agent; live tests use InMemorySaver (no writes to main)
- Open issues: cold start on Render to re-measure (T13; warm-up in background if slow; keep awake during review?); conversations with personal details kept in Neon with no retention; prompt-example test catches copies, not paraphrases; 5 kW reply still doesn't name HomePro 5 (T12); live test budget 24 calls vs ~15 target; still open from T7: reason latency (T8b), session_id race; T6: buildFilter unseen, URL private until T9, client IP behind proxy (T9), checksPass + CORS (T13); T4: keyword-only fallback gaps, word stats cached per process, same questions for tuning and reporting, "Dehli" typo (T12)
- Next: T8b lead tools (create_lead, update_lead, schedule_callback, create_support_request), flows F2-F4, demo script → HOT lead

## T8b: lead tools and lead-aware conversation (2026-10-11)
- Status: done
- Changed: app/agent/leads.py (progress block, city confirmation, write-once sync, lead summary), 4 lead tools in tools.py (no arguments, InjectedState, no score in results), graph (new understand fields, retrieval gate, support_case guard, sync, lead summary), /api/chat (lead summary, CHAT_SHOW_LEAD_SCORE, 64 striped session locks with 60 s timeout, background refresh_reason), leads service (race on session_id, reason only on score/status change, reason_now, refresh_reason), app/core/tracing.py + LANGSMITH_HIDE_* settings, prompts (system, rag, lead_extraction), render.yaml + .env.example, 3 new test files, PLAN, ARCHITECTURE, EVAL
- Tests: `cd backend && python -m pytest -q` → 379 passed, 0 failed (12 live deselected); live: test_live_lead_flow 2 passed (twice for the demo), test_live_chat q01 + Pune 2 passed
- Also ran: live demo on the test branch, 4 turns → lead 97 HOT (listed), LLM reason written after the reply, action "Contact the lead within 1 hour… Customer asked for a callback: tomorrow evening", callback requested; turn times 7.5 / 5.8 / 6.6 / 6.4 s, LLM calls 2 / 3 / 3+1 / 3; "Yes please, connect me" → "Could you please share your name and phone number?" (no search); reason cost measured: 3,569 and 3,359 ms with the LLM vs 319 and 313 ms with the template; `eval_retrieval --rewrite` once: rewrite hit@3 23/24 (was 24/24), q09 1 → 7, off-topic 8/8; unit suite makes 0 external HTTP calls; LLM calls: live 36, eval 32, timing 2
- Prompt changes: lead_extraction (contact_request, gives_details, asked_city, callback_time, issue, project_details, declines); rag_prompt (lead progress block); system_prompt T8b version (contact first, one question up to two details, city confirmation, gentle next step, tool rules, no score/status, claims only after a tool; replaces "no tool yet"), then "team can follow up, offer a callback" (live run 1 asked about shade and said "will follow up")
- Decisions: reason written after the reply (it added about 3.2 s); new details written by the update_lead tool or the finalize sync, never both (snapshot); no KB search for details-only or contact-request turns; city confirmed in code only when the assistant's question named it and the reply doesn't say no; status/score in the API only with CHAT_SHOW_LEAD_SCORE (demo on, production off); live tests write leads only to the test branch
- Open issues: understand slower (2.4-4.1 s, was 1.2-1.7 s) with the bigger schema; rewrite eval 23/24 (Pune relies on check_service_area); live budget 36 vs ~30; tracing privacy and CHAT_SHOW_LEAD_SCORE=false for production (T13); still open: cold start on Render (T13), conversation retention, prompt-example test catches copies not paraphrases, 5 kW reply without HomePro 5 (T12), buildFilter unseen, URL private until T9, client IP behind proxy (T9), checksPass + CORS (T13), keyword-only fallback gaps, word stats cached per process, "Dehli" typo (T12)
- Next: T9 Sales API, API key, rate limit, error paths, API tests

## T9: sales API, API key, chat rate limit (2026-10-11)
- Status: done (local; the Render live check in docs/DEPLOY.md is for Adithya after the push)
- Changed: app/api/sales.py (5 sales endpoints via the T7 service), app/api/security.py (client IP, chat limit, sales key), app/core/rate_limit.py, main.py (error handlers, CORS, ip_source log), chat.py (UUID v4 session ids, rate limit), service.list_support_requests + SupportRequestView, config + render.yaml + .env.example (CLIENT_IP_HEADER, CHAT_LIMIT_*), 3 new test files, session ids in 4 test files, PLAN, ARCHITECTURE, DEPLOY
- Tests: `cd backend && python -m pytest -q` → 444 passed, 0 failed (12 live deselected; was 387); live tests collected, not run (no LLM calls this task)
- Also ran: local uvicorn with a throwaway key: /docs 200 with the X-API-Key scheme and 8 paths; /api/leads no key 401, key 200, /api/leads/999999 404; chat session_id "demo-1" 422; empty messages 422 ×9 then 429, Retry-After 47, friendly detail; log ip_source "fallback", no key or client IP in the log
- Prompt changes: none
- Decisions: client IP from CF-Connecting-IP only (Render + Cloudflare docs: Cloudflare overwrites it, only appends to X-Forwarded-For; uvicorn "*" takes the spoofable leftmost XFF), IPv6 per /64, fallback = connection address; chat 10/min + 100/day per IP (Adithya), every request counts, no global cap, in memory for one instance; key: constant time, fail closed (unset or < 16 chars → 503), 10 wrong keys / 15 min / IP → 429, missing key not counted; session_id never returned by the sales API; every 422 is {"detail", "errors"} without echoing input; POST /api/leads refuses any invalid field (422) and writes the LLM reason after the response
- Open issues: Render live check pending (Environment vars, ip_source "header" in the logs); T14: how reviewers get the sales key (never README/repo/video), rotate after review; T10: keep the returned session_id, show 429 detail and wait Retry-After; still open: cold start on Render (T13), conversation retention, 5 kW reply without HomePro 5 (T12), checksPass + CORS Vercel URL (T13), tracing privacy and CHAT_SHOW_LEAD_SCORE=false for production (T13), buildFilter unseen, keyword-only fallback gaps, word stats cached per process, "Dehli" typo (T12)
- Next: T10 Chat page (welcome + starters, messages, source chips, steps panel, lead card, error states)

## T10: customer chat page (2026-10-12)
- Status: done
- Changed: frontend/ (React 19 + Vite 8, plain JS/CSS: api.js, storage.js, useChat, useWelcome, App, components ReplyText/SourceChips/StepsPanel/LeadCard/Composer/Message, styles.css, favicon, 6 test files; exact pins + lockfile), backend/scripts/serve_test_branch.py + test, PLAN (3.13, 3.14, T9/T10 rows, T11-T13 notes), ARCHITECTURE, DEPLOY (T9 check fixes, local check, Vercel), CLAUDE.md commands
- Tests: `cd backend && python -m pytest -q` → 448 passed (12 live deselected; was 444); `cd frontend && npm test` → 40 passed; `npm run build` ok; npm audit 0 vulnerabilities
- Also ran: headless Edge (playwright-core in scratchpad) against serve_test_branch, VITE_API_URL with a trailing slash: welcome 0.9 s; turn 1 KB Section 8 tags + chip (7.9 s) → Lucknow listed, asks name + phone (5.2 s) → lead #1 HOT 97 (4.8 s) → callback requested (3.8 s); 12 LLM calls; reload kept session + 9 messages; 400 px no horizontal scroll; dark mode checked; /api/leads via launcher → only the test-branch lead
- Prompt changes: none
- Decisions: JS + JSDoc; sessionStorage (id, transcript, lead); chips open on click; citations as "KB Section N" tags; default API 127.0.0.1:8000 (a Docker/WSL relay answers 404 on localhost/::1:8000); launcher also points the sales API (leads_engine) at the test branch; textarea read-only while waiting; 3 s waking notice, 8 s slow-answer note, 90 s timeout
- Open issues: turn 4 reply ended "[schedule_callback result]" (prompt fix, T12); demo card shown to customers in production? (T13); VITE_API_URL + Vercel URL in CORS_ORIGINS (T13); Lighthouse not run; still open: cold start on Render (T13), conversation retention, 5 kW reply without HomePro 5 (T12), checksPass (T13), tracing privacy + CHAT_SHOW_LEAD_SCORE=false for production (T13), buildFilter unseen, keyword-only fallback gaps, word stats cached per process, "Dehli" typo (T12)
- Next: T11 Dashboard (API-key prompt, leads table + filter, lead detail with breakdown)

## T10b: fast understand and a natural voice (2026-10-12)
- Status: done
- Changed: app/core/llm.py (understand_llm: main → fast → Groq, each cut off at 5 s by with_deadline, 8-worker pool; GeminiChat.request_http), graph (understand_llm, city source = KB Section 10 summary, citation safety nets), leads.status_block (info-only line), answer.py (strip_tool_citations, tidy_citations), understand step reports its model, eval_retrieval --rewrite on understand_llm, prompts (system, rag, README), tests (new test_live_voice.py; llm, graph, leads, answer; live Pune assert, paced live turns), PLAN, ARCHITECTURE, EVAL, CLAUDE.md
- Tests: `cd backend && python -m pytest -q` → 471 passed, 0 failed (13 live deselected; was 448); live: test_live_voice 1 passed, demo 1 passed, Pune 1 passed, q01 1 passed (after a fix)
- Also ran: understand timing, same input 3× each: fast 17.4 s median (worst 18.5 s), main 1.7 s (1.9 s), Groq 1.0 s (1.05 s, about 2,300 tokens a call); Gemini refuses deadlines under 10 s (400), so HTTP 10 s + the 5 s wall-clock guard; before → after (final run): Adithya's 4 turns 22.1/13.8/9.7/32.9 s → 5.0/4.3/4.9/5.9 s, understand 6.7-29.7 s → 1.3-1.8 s; demo 17.6/17.2/49.4/18.9 s → 4.1/3.8/18.0/4.1 s (turn 3 steps 4.7 s, the rest is the background reason TestClient waits for), lead 97 HOT, callback requested; `eval_retrieval --rewrite` once: rewrite hit@3 24/24 (was 23/24), hit@1 23/24, MRR 0.979, all 32 on the main model; LLM calls about 184 answered + 18 rejected (400), over the ~110 budget (encoding crash, one run before the 10 s finding, 2 city-wording rounds, q01 fix)
- Prompt changes: system_prompt (plain words with names kept exact, numbered lists, each KB section cited once with its full label and never a tool, an offer after information questions, details only after interest, unlisted city explained once in its own words then "same goes", never "doesn't serve"; safety rules unchanged); rag_prompt (cite once, never a tool); progress block info-only line (code)
- Decisions: understand on the main model (fast too slow, Groq's 8K tokens a minute too low), fallbacks fast then Groq, 5 s each; chat limits stay 10/min + 100/day (Adithya) although a turn now uses about 2.75 main calls (about 55% of the day's quota per IP); citation safety nets in code (strip tool names, tidy labels); a city answer's chip is the check_service_area summary chunk
- Open issues: lead reason still on the slow fast model (14-30 s, background only); a main-model timeout costs 5 s more on the slow fast model before Groq; numbered lists show one paragraph per line (page redesign); "reference locations" wording still slips in; still open: demo card in production? (T13), VITE_API_URL + Vercel URL in CORS_ORIGINS (T13), Lighthouse not run, cold start on Render (T13), conversation retention, 5 kW reply without HomePro 5 (T12), checksPass (T13), tracing privacy + CHAT_SHOW_LEAD_SCORE=false for production (T13), buildFilter unseen, keyword-only fallback gaps, word stats cached per process, "Dehli" typo (T12)
- Next: chat page redesign (Adithya), then T11 Dashboard

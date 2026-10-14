# ARCHITECTURE.md

Short map of how the code is built. Updated at the end of each task. The overview is in the [README](../README.md); `docs/PLAN.md` is the design record written while building.

## Backend modules (as of T13e)
| Module | Role |
|---|---|
| `app/main.py` | FastAPI app, startup hook (`init_db`), request-id + logging middleware (skips `QUIET_PATHS`: `/api/health/live`; adds `ip_source` for rate-limited requests, never the IP), error handlers (`ApiError`, 422 with one message per field, database down → 503), CORS (GET/POST; `Content-Type`, `X-API-Key`, `X-Request-ID`; exposes `Retry-After`), routers |
| `app/core/config.py` | `Settings` (pydantic-settings, reads `backend/.env`), `get_settings()` |
| `app/core/rate_limit.py` | `RateLimiter(limits)`: in-memory fixed windows per key, thread-safe; `hit()` counts or returns the seconds to wait (a refused request is not counted), `blocked()` only checks |
| `app/api/security.py` | `client_ip` (the `CLIENT_IP_HEADER` value if it is one valid IP, else the connection address; never `X-Forwarded-For`), `limit_key` (IPv6 per /64), `chat_rate_limit` (10/min + 100/day per IP → 429 + Retry-After), `require_sales_key` (`X-API-Key`, constant-time, fails closed, 10 wrong keys per 15 min per IP → 429), `ApiError` + `error_response` (the JSON error shape) |
| `app/api/sales.py` | Sales router (key required): `GET /api/leads`, `GET /api/leads/{id}`, `POST /api/leads`, `POST /api/callback` (T12a: 201 new, 200 when the lead's request gets the new time), `GET /api/support-requests`, all through `app/leads/service.py`; never returns `session_id`. T11: `GET /api/leads` also takes `q` (name or city) and returns `counts` (leads per status in the whole table); its items carry `customer_type`, `property_type` and `source_channel` |
| `app/core/logging.py` | JSON log formatter, `request_id_var` |
| `app/core/db.py` | `Base`, `get_engine()` (Neon, pre-ping), `check_db()`, `init_db(engine)` (pgvector + `create_all`) |
| `app/core/llm.py` | `embedder()` and `query_embedder()` (google-genai, one Content per text, batches of 20, LRU cache for queries; T12a: a query is one attempt cut off after 3 s, and the embedder is skipped after 2 failures in a row). `GeminiChat`: adds its `request_http` (10 s, its minimum, and no retry) to every Gemini request. T12a: `call_with_deadline` (8-worker pool), `failure_kind` (timeout, 429, 5xx), `SkipList`/`SKIPS` (a model passed over while its window is open), `Candidate` and `model_chain` (one attempt each, skip, log, `LLMError` when all fail). **Now (T13d-T13e):** `understand_llm(schema)`: main, then fast, 5 s each (no Groq). `chat_llm(model, tools=, schema=)` (`answer_candidates`): main 8 s, Groq `FALLBACK_MODEL` 10 s, Groq `FALLBACK_MODEL_2` 10 s, fast 8 s (Groq: `parallel_tool_calls=False`); its `last_try` (`last_try_chain`) waits up to 10 s for a paused model or gives the main model 20 s. `reason_llm()`: fast only, 8 s, after the reply. `turn_scope`/`TurnBudget`: every model call of a chat turn fits in 40 s (3 s kept for finalize), none starts with under 2 s left, and a model that timed out is skipped for the rest of the turn. Skip windows (`SKIP_SECONDS`): a timeout or 5xx pauses the model 15 s for that use only, doubling to 30 and 60 s on a repeat within 30 s (`BACKOFF_GRACE_S`); a 429 pauses every use for the provider's retry-after (`retry_after_s`), else 60 s; one failure opens a window, the embedder needs 2. `GEMINI_SERVICE_TIER`. Superseded (T12a-T13c): understand main, Groq, fast at 4 s; answer main 8 s, Groq 10 s, fast 8 s; reason main only, 4 s; windows 60 s after a timeout or 429, 120 s after a 5xx. Still current: T12b, offline scripts only (eval, seed): `Embedder.embed_queries` (batches of 20, ingest's 30 s and 3 attempts, no skip list), `NoSkips`, `patient(chain)` (the whole chain retried after a timeout, 429 or 5xx, 5 s then 20 s apart), `offline_understand_llm(schema)` and `offline_reason_llm()` (30 s per call); the chat stays fail-fast. `main_llm()`, `tokens_of()`, `invoke_llm()`, `provider_of()`, `reply_text()`. T13c: `skip_scope` (a timeout skips the model only for the use that timed out, a 429 or 5xx for every use; `SkipList.left(key, use)`), `error_detail` (the provider's status and error type or code, identifiers only, in "llm call failed"), `AFC_OFF` on every Gemini request, and LangSmith runs via `traceable` (`_attempt` per model attempt, `_skipped`, `_traced_embedding`, `note_failure`; `model_chain:<use>` names the chain) |
| `app/core/prompts.py` | `load_prompt(name)`: reads `prompts/<name>.md` (cached) |
| `app/agent/state.py` | `AgentState` (persisted `messages`, `lead`, T12a `ruled_out`; per-turn fields, T12a: `new_details`, `declines`, `lead_created_now`, `callback_now`; T12b: `area_checks`; T13a: persisted `city_source`, and `asked_cities` now holds the open candidates) and `turn_input()`, which resets the per-turn fields |
| `app/agent/understand.py` | `Understanding` schema (intents incl. T12a `asks_what_is_needed`, `search_query`, details generated from `LeadData`, T12b `cities`: every city named in the message), `load_kb_hints` (section titles + table summary chunks from `kb_chunks`; T13a: `guidance`, the KB Section 9 rows, added to the context on a turn that gives the monthly bill; T13b: only on a package turn whose property type is known), `understand()` (also returns the call's tokens). T13e: the AI understands, the code enforces: `reply_type` (answers, skips, accepts, declines, goes_ahead, closes, asks_new), `ruled_out`, `locality` and `place_relation` (code confirms only a place inside a listed city), `asks_offerings`, `problem_now` (only "yes" with an issue starts the support flow); the assistant's last message in full (`LAST_REPLY_CHARS`); word lists in `text_details.py` run only when this step fails |
| `app/agent/questions.py` | T13e: what a reply may ask or offer, read from our own replies: `ask_rules` (no near-repeat of the last question, each offer kind once per chat, none while parked, no callback ask for an explorer, no contact ask on an info-only turn, one detail per question), `fix_region_reply` (a KB region is never confirmed: the fixed line with its cities, then which city), `last_question_kind` (backup only) |
| `app/agent/tools.py` | `make_tools`: `search_knowledge_base`, `check_service_area` (content + KB chunk artifacts) and, with `LeadOps`, `create_lead`, `update_lead`, `schedule_callback` (T12a: a new time updates the lead's one request, `"updated": true`), `create_support_request` (no arguments; read the draft via `InjectedState`; no score in results); `tool_step` (T12b: a saved support request's step shows its id); T12b `service_area_data` (the city result, shared with the graph's own check) |
| `app/agent/graph.py` | `build_graph(AgentDeps)`: understand → retrieve → agent ⇄ tools → finalize, plus welcome / closing / redirect; `build_agent()`, `get_agent()` (FastAPI dependency). T12a: understand merges callback times and short answers and turns a time of day into a callback time; finalize runs the reply guards (`strip_unsourced_citations`, `remove_false_claims`, `fix_callback_claims`, `soften_promises`), each a "guard" step when it fires; steps carry the model and tokens. T12b: understand turns a KB question labelled `asks_what_is_needed` into a search (`own_details_question`) and searches with the message when a question came with no query; `check_named_cities` checks every city named before the model (an `area_check` step, the KB Section 10 summary as the first chunk, the results in the agent's message); retrieve adds the catalog summary for a stated size (`add_catalog_for_size`); finalize adds the number guard (`remove_unsupported_numbers`), `plain_service_area_words` and `keep_one_question`; a model failure after a tool wrote something replies with what was done (`tool_outcome_reply`). T13b: `package_guidance` (the `<package_guidance>` block from `prompts/package_guidance_prompt.md` on a package turn, right after the context), and the KB Section 9 rows only when `package_case` says home or business. T13c: understand uses `place_type` and drops a new type from a passing mention (step `dropped`) and adds `chat_intent` (step `intent_from_chat`); retrieve adds the missing KB Section 9 rows when part of the package list was found (`add_all_packages`, step `added_package_rows`); after create_lead answers "saved": false the turn goes on with `agent_llm_without_create` (`create_lead_refused`, step `create_lead_offered: false`); finalize adds `remove_uncovered_policies` and `all_or_no_packages`, and a reply the guards left empty becomes `NUMBERS_CONFIRM_TEXT` (never a blank bubble). T13d: understand reads the what-if and look-alike-city cases; when every answer model fails the agent node calls `last_try` once (only if no tool wrote anything), then sets `model_failed` and finalize answers with the safety net (`fallback_reply`, step `fallback`); finalize saves a ready lead in code when the model didn't call create_lead (`save_ready_lead`, step `save`, `add_saved_line`); new guards `remove_free_claims`, `drop_detail_questions` (no third ask after two skipped ones) and `add_missing_tag`; `answered_by` (model, knowledge_base, tools, fixed). T13e: routing, the skip count and parked asks come from the understand step's `reply_type`; places are confirmed only inside a listed city (`place_relation`); support starts only on `problem_now` yes with an issue; the KB Section 8 catalog comes first for an offerings question (`add_offerings_line`); `add_size_line` (a site assessment confirms the final size); `plain_kb_words`, `remove_kb_rules`, `fix_region_reply` and `ask_rules` (`app/agent/questions.py`) |
| `app/agent/leads.py` | `LeadOps` (leads database, reason model, `reason_now`), the lead progress block (`status_block`; T10b: says not to ask for details until `shows_interest`; T12a: after the save, the details that would still add points and one next line, with `next_detail` chosen in code), `confirmed_city` (T13a: only when the question named exactly one candidate), T13a `resolve_city` (an asked city counts as the customer's; two or more in question: none guessed, they become candidates) and `drop_known_city_question` (removes a question for a city the draft already has; the sentences go into the guard step), T12a `short_answer_details`, `merge_callback_time`, `is_clock_time`, `ruled_out_details`, `remove_false_claims`, `fix_callback_claims`; T12b `own_details_question`, `keep_one_question` (the question for the next missing detail, else the last; T13b: "home or business?" on a package turn whose type is unknown); T13b `package_case` (a turn that gives the bill, or the property type after it, with no size settled: "home", "business" or "unknown"; None in a support case or on a contact request); T13c `place_type` (replaces T13b's `customer_type_from_words`: home words → residential, business words → commercial, only as a short answer to the bot's home-or-business question or in a phrase about their own place, never from a passing mention; no intents needed; the graph shows it as `from_words`), `passing_place_kinds` (for the type guard) and `chat_intent` ("interested" from buying interest or a callback while the intent is unknown); `pending_changes` (write-once sync), `lead_summary` for the API (T10c: plus `card_fields`: name, city, property and customer type, and the names of the known fields, never their values) |
| `app/agent/fallback.py` | T13d, the safety net: `fallback_reply` when no model answered a turn, built only from what needs no model: the code's city checks, one or two KB sentences from this turn's chunks that share a key word with the message (an FAQ answer first, never a sentence written for the assistant: `is_instruction`), what this turn saved, and one question or a fixed offer; never "I'm having trouble" |
| `app/agent/text_details.py` | T13d-T13e, backup only (a turn whose understand step failed): `details_from_text` (phone, email, city, bill, size, a name from clear patterns, clear intent and timeline phrases), `short_reply`, `reply_type_from_words`, `is_go_ahead`, `asks_to_be_called`, `asks_what_is_offered` (T13d's what-if word checks were replaced in T13e by the understand step's `problem_now`) |
| `app/agent/packages.py` | T13c, reference packages all or none (no size settled): `reference_packages` (names and kind from the KB Section 9 rows), `names_in`, `size_settled`, `SIZE_RE` (moved from graph.py), `add_all_packages` (the missing Section 9 rows join the context when the search found part of the list), `all_or_no_packages` (a partial list leaves the reply, never a package the customer named in this message; quoted in the guard step) |
| `app/core/tracing.py` | `configure_tracing`: optional LangSmith; copies LANGSMITH_* settings into os.environ only when tracing is on and a key is set (the runs themselves: `app/core/llm.py`, T13c) |
| `app/agent/memory.py` | `get_checkpointer()` (PostgresSaver over a psycopg pool), `setup_checkpointer()` (never fails the boot) |
| `app/agent/welcome.py` | Fixed texts: welcome (F0), starter questions, closing line, off-topic redirect; T12a: the honest lines the guards use (still needed, ready to pass on, callback logged / updated / exists, ask for a call time); T12b: `NUMBERS_CONFIRM_TEXT` (the team confirms exact numbers), `SAVED_FALLBACK_TEXT`, `SUPPORT_SAVED_FALLBACK_TEXT`; T13c: `UNCOVERED_TEXT` (can't confirm, the sales team can tell), `PACKAGES_FIT_TEXT` |
| `app/api/welcome.py` | `GET /api/welcome` |
| `app/api/health.py` | `GET /api/health` (status, db, kb_chunks); `GET /api/health/live` (`{"status":"ok"}`, no DB, Render's health check) |
| `app/api/chat.py` | `POST /api/chat`: rate limited (`chat_rate_limit`); `session_id` must be a UUID v4 (the server issues one when none is sent); validates the message, runs the agent with `thread_id` = `session_id`, returns reply, sources, steps, `lead` (summary; status and score only with `CHAT_SHOW_LEAD_SCORE`) and degraded. One turn per session (striped locks, 60 s timeout); a pending lead reason is written after the reply (background task). DB errors → 503 (T12b: "We can't reach our database right now…", also for a failed lead write); anything else → 200 degraded. T13d: the turn runs inside `turn_scope()` (40 s for model calls); the response carries `answered_by`; a repeated `client_message_id` (the page resends once when no reply arrived) gets the reply already sent (`RecentReplies`, 500 replies, 10 minutes, in memory) |
| `app/rag/answer.py` | Helpers the agent reuses: `format_context` (`[KB Section N: Title]` labels), `rag_message`, `cited_sections`, `pick_sources` (one per cited section; T12a: none when nothing is cited), `normalize_citations`, `strip_tool_citations` (removes a tool cited like a source), T12a `strip_unsourced_citations` (tags only on facts from this turn's chunks; never on a question, a thank-you or a claim), `is_claim`, `split_sentences`, `soften_promises` ("will follow up" → "can"), T12b `plain_service_area_words` ("reference locations" → "service areas", outside tags), `rag_message` with the city checks (T13b: and the package rule after the context, its line removed on other turns), `tidy_citations` (each KB section cited once, with its full label), `retrieve_step`, `LLM_DOWN_REPLY`; T13c `remove_uncovered_policies` (a sentence stating something about discounts, coupons, cashback, promotions, special offers or brands that no chunk of the turn mentions becomes one honest line) |
| `app/rag/keywords.py` | T13d: `key_words` of a text (with a synonym and Hinglish map for matching only, never a fact), used by the safety net to pick a KB sentence and by `add_missing_tag` to find the chunk a reply used |
| `app/rag/numbers.py` | T12b number guard: `number_mentions` (digits with ₹/k/lakh and units, number words and vague amounts with a unit), `known_numbers`, `is_supported`, `remove_unsupported_numbers` (a sentence stating a number no allowed text gives is removed and, T13a, quoted in the guard step, never in the log; for a duration, amount or percentage the fixed "team confirms" line takes its place once). Allowed: this turn's chunks and tool results, every customer message and earlier reply, validation messages |
| `app/rag/pdf_parser.py` | `parse_kb(path)`: KB PDF to numbered sections with text lines and tables (pdfplumber; cover, header and footer removed) |
| `app/rag/chunker.py` | `Chunk`, `chunk_sections()`: one chunk per table row (KB Sections 8-12) plus one `table_summary` chunk listing every row (8, 10), one per FAQ item (15), prose split at ~1,200 chars, Sections 16-19 tagged `policy`; `embed_text()` = `title: ... \| text: ...` |
| `app/rag/models.py` | `KbChunk` (`kb_chunks`: vector + generated tsvector with GIN index) and `ServiceArea` (`service_areas`, unique on city + region) |
| `app/rag/ingest.py` | `python -m app.rag.ingest`: parse, chunk, embed all first, then replace all rows in ONE transaction; `--dry-run` prints chunks and touches no DB or key |
| `app/rag/retriever.py` | `search(query, k, mode)`: keyword (stemmed words OR'd, words in more than 30% of chunks dropped, `ts_rank_cd`) + vector (cosine), RRF fusion (k = 60, keyword weight 0.5), relevance gate (similarity only when a vector search ran), keyword fallback with `degraded`; policy chunks never returned |
| `app/leads/schemas.py` | `LeadData` (the lead contract: every field optional, null = unknown, units in names, per-field `description` for T8's structured output, `from_raw` keeps valid values and reports invalid ones), `LeadView` and the result models |
| `app/leads/validators.py` | Pure validators: Indian mobile → `+91XXXXXXXXXX`, email syntax (email-validator, no network), ₹ amounts (`₹4,000`, `4k`, `1.2 lakh`; `AMOUNT_MULTIPLIERS` shared with the number guard), kW, kWh, names, text; plain error messages (T12b: `PHONE_INVALID`) |
| `app/leads/scoring.py` | `score_lead` (rubric, caps at WARM and 74, recorded only when they lower the score; T13d: the exploring cap at COLD and 44, and a call request always counts as ready; breakdown JSON), `status_for`, `recommended_action` (T12a: a callback time replaces the status's timing), `missing_details` (most useful first), T12a `next_status_gap`, T12b `status_gaps` (with each gap's rule, so the template never names a factor twice), `missing_for_creation` |
| `app/leads/reason.py` | `write_reason`: one sentence from `reason_llm` (T13d: the fast model, 8 s, after the reply; T12a had the main model; breakdown and next-status gap only, no personal details or money), template fallback; `gap_phrase` ("not HOT yet: no timeline"), `problem_with`. T12b: the LLM writes only the first part and code adds the ending once; a reply that copies the prompt, names another status or repeats the ending gets the template |
| `app/leads/models.py` | `Lead` (`leads`), `Callback` (`callbacks`, status `requested`), `SupportRequest` (`support_requests`) |
| `app/leads/service.py` | `create_lead` (refuses until the minimum fields, one lead per session; channel `web`, `whatsapp` or `demo`), `update_lead` (merge, re-locate, re-score), `get_lead`, `list_leads` (status, `search` in the name, city or the customer's city words with LIKE wildcards escaped, sort, paging), `lead_counts` (one GROUP BY), `schedule_callback` (T12a: one requested callback per lead, a new time updates it), `create_support_request`, `list_support_requests`; logs ids, status and score only |
| `app/services/service_areas.py` | `ServiceAreas.lookup(city, state)`: alias map + rapidfuzz (score 85), then (T12b) one swapped pair of letters (`one_swap_apart`: "Dehli" → Delhi; never any other one-letter change, so Raipur stays unverified); loads from the `service_areas` table (`get_service_areas()`, cached). T13d: separate places ("Greater Noida") and landmark phrases ("near Lucknow") make a listed name unverified. T13e: `lookup` checks the city alone (the understand step puts a locality in its own field), a KB region ("Delhi NCR", "UP") is a region and never a city (`regions`, `region_around`), and the phrase lists run only as the backup when the understand step failed (`find_in_text`) |
| `scripts/list_models.py` | Lists Gemini chat and embedding models for the key |
| `scripts/search_kb.py` | Prints the mode, `no_context` and the top chunks with scores and ranks |
| `scripts/serve_test_branch.py` | Dev only: runs the API on 127.0.0.1:8000 with the chat agent's lead writes and the sales API (`leads_engine`) on `TEST_DATABASE_URL`; KB and service areas read from main; chat memory in the process (`InMemorySaver`). `engine_for_test_branch` refuses an empty or same-as-main URL. T12a: `--slow-gemini` (`force_gemini_timeouts`, `HangingGemini`) makes every Gemini chat call hang 10 s for the slow-Google drill; a flag of this dev script only, not in the Docker image. T12b: `--port` |
| `scripts/seed_demo_leads.py` | T11: 8 demo leads (source_channel `demo`) through `service.create_lead`, 2 callbacks and 1 support request, clearly fake contacts; skips what is already there; test branch by default (`engine_for_test_branch`), `--main` only after "main" is typed. T12a: `--reset` (test branch only) empties the lead tables first; the summary prints each reason. T12b: reasons on `offline_reason_llm` |
| `scripts/eval_retrieval.py` | Embeds the 24 KB and 8 off-topic questions once, runs `search()` in all three modes (k = 10), prints hit@1, hit@3, MRR, the misses and the gate results, and writes `docs/EVAL.md` (its "Tuning notes" section is hand-written and kept). `--rewrite` adds the understand step's rewritten query (`rewrite` and `fused` modes) and the small-to-big report. T12b: also the 8 held-out questions (`data/eval/holdout_eval.jsonl`) in their own section, "best section" and "any listed section" side by side; embeds and rewrites on the offline path |
| `scripts/live_scenarios.py` | T12b, dev only: the held-out chat and the assignment's edge cases against a running `serve_test_branch` (`--group holdout`, `edge` or `slow`), one session per scenario, PASS/FAIL per check, a markdown table for `docs/EVAL.md`; the number check reads the KB from main (read-only). T13c: `--group replay` (C8, chat 1 of Adithya's break-it test, then the discount question) with `no_question_for_known`, `names_all_or_none`, `tool_ran_at_most_once`, `callback_requested`, `no_policy_on` |

## Request flow
```mermaid
flowchart LR
  client["Client"] --> mw["Middleware: request id, JSON log, 500 guard"]
  mw --> live["GET /api/health/live (no DB)"]
  mw --> health["GET /api/health"]
  health --> check["check_db: SELECT 1"]
  health --> count["count_kb_chunks"]
  check --> neon[("Neon Postgres + pgvector")]
  count --> neon
  mw --> welcome["GET /api/welcome (fixed text)"]
  mw --> limit["chat_rate_limit: 10/min, 100/day per client IP<br/>(CF-Connecting-IP on Render)"]
  limit -.->|"over"| e429["429 + Retry-After, friendly detail"]
  limit --> chat["POST /api/chat (session_id: UUID v4)"]
  chat --> agent["agent graph (see Agent graph)"]
  agent --> neon
  agent -.->|"DB error"| e503["503, friendly message"]
  mw --> key["require_sales_key: X-API-Key, constant time"]
  key -.->|"missing / wrong; 10 wrong per 15 min → 429; key unset → 503"| e401["401 / 429 / 503"]
  key --> sales["/api/leads, /api/leads/{id}, /api/callback,<br/>/api/support-requests"]
  sales --> service["app/leads/service.py (T7)"]
  service --> neon
```

## Ingestion (T3)
```mermaid
flowchart LR
  pdf["KB PDF (local)"] --> parse["pdf_parser: sections + tables"]
  parse --> chunk["chunker: table rows, FAQ, prose, policy"]
  parse --> areas["service_areas: KB Section 10 cities"]
  chunk --> embed["embedder: gemini-embedding-2, 768 dims"]
  embed --> tx["one transaction: replace rows"]
  areas --> tx
  tx --> db[("kb_chunks + service_areas")]
```

## Retrieval (T3, tuned in T4)
```mermaid
flowchart LR
  q["question"] --> words["stemmed words"]
  words --> filter["drop words found in more than 30% of chunks (stats cached)"]
  filter --> kw["keyword search: top 10, OR query"]
  q --> emb["embed_query (one attempt, 3 s; skipped after 2 failures in a row)"]
  emb --> vec["vector search: top 10"]
  emb -.->|"fails"| fb["keyword only, degraded = true"]
  kw --> rrf["RRF fusion: keyword weight 0.5, vector 1, top k"]
  vec --> rrf
  vec --> gate["relevance gate: best similarity below 0.63 means no_context"]
  fb -.->|"keyword hits count"| gate
  kw --> db[("kb_chunks")]
  vec --> db
```

## Retrieval eval (T4)
```mermaid
flowchart LR
  qs["24 KB + 8 off-topic questions"] --> embed["embed each once"]
  embed --> run["search() in keyword, vector and hybrid mode, k = 10"]
  run --> metrics["hit@1, hit@3, MRR, misses, gate counts"]
  metrics --> md["docs/EVAL.md, hand-written notes kept"]
```

## Deployment (T6)
Files at the repo root: `Dockerfile` (python:3.12-slim, only the app package and `prompts/`, non-root user, one uvicorn worker on `$PORT`), `.dockerignore`, `render.yaml` (one free web service, region Singapore, next to Neon). Steps and limits: `docs/DEPLOY.md`. T13b: the frontend is on Vercel (<https://vantis-sales-agent.vercel.app>, `frontend/vercel.json`), and Render's `CORS_ORIGINS` is that one URL. T14: `.github/workflows/keep-warm.yml` calls `/api/health/live` every 10 minutes from 08:00 to 23:00 IST during the review (switch-off steps in the file and in DEPLOY.md).
```mermaid
flowchart LR
  dev["git push to main"] --> gh["GitHub repo"]
  gh -->|"buildFilter: Dockerfile, backend/app, requirements, prompts"| build["Render: docker build (free, Singapore)"]
  build --> web["Container: uvicorn app.main:app on $PORT"]
  render["Render health check, every few seconds"] --> live["/api/health/live (no DB, not logged)"]
  live --- web
  web -->|"DATABASE_URL"| neon[("Neon Postgres + pgvector, AWS Singapore")]
  web -->|"GOOGLE_API_KEY / GROQ_API_KEY"| llms["Gemini, Groq"]
```

## Leads layer (T7)
Called by T8's tools and T9's sales API (`app/api/sales.py`). Rules in plain words: `docs/SCORING.md`.
```mermaid
flowchart LR
  raw["raw details (agent or API)"] --> parse["LeadData.from_raw: validators, errors kept apart"]
  parse --> min{"name + phone/email + city + a need?"}
  min -->|"no"| missing["missing_fields, nothing saved"]
  min -->|"yes"| loc["ServiceAreas.lookup: listed / unverified"]
  loc --> score["score_lead: 6 rules, caps, HOT / WARM / COLD"]
  score --> reason["write_reason: reason_llm (fast, 8 s, after the reply) or template"]
  score --> action["recommended_action"]
  reason --> db[("leads, callbacks, support_requests")]
  action --> db
```

## Agent graph (as of T13e)
One customer message; state checkpointed in Postgres per `session_id`. The README's "How it works" walks through it; PLAN.md part 3.1 has the history of each rule.
```mermaid
flowchart TD
  msg(["POST /api/chat, inside turn_scope (40 s for model calls)"]) --> understand["understand<br/>main, then fast (5 s each), structured output:<br/>intents, details, cities, reply_type, problem_now, search_query<br/>(both fail: raw message + text_details backup)"]
  understand -->|"anything else"| areas["area check in code, every city named<br/>area_check step, KB Section 10 source<br/>(+ catalog for a size or an offerings question,<br/>+ Section 9 rows on a bill turn)"]
  understand -->|"only hello / what can you do"| welcome["welcome text"]
  understand -->|"only thanks / bye"| closing["closing line"]
  understand -->|"only off-topic, no query"| redirect["polite redirect"]
  areas -->|"search_query, not details-only"| retrieve["retrieve<br/>hybrid search, top 4<br/>(+ all package rows when part was found)"]
  areas -->|"details or contact request only"| agent
  retrieve --> agent["agent<br/>main 8 s → Groq 10 s → Groq 2 10 s → fast 8 s, tools bound<br/>(no create_lead after it could not save)<br/>all fail, no tool wrote: one last try"]
  agent -->|"tool calls (max 4 rounds)"| tools["tools<br/>search_knowledge_base, check_service_area<br/>create_lead, update_lead,<br/>schedule_callback, create_support_request"]
  tools -.->|"lead tools (test branch in live tests)"| leadsdb[("leads, callbacks,<br/>support_requests")]
  tools --> agent
  agent -->|"answer, or model_failed"| finalize["finalize<br/>code-side save when ready (save_ready_lead),<br/>reply guards, sources, steps,<br/>write-once lead sync, lead summary"]
  finalize -.->|"no model answered"| net["safety net (fallback.py):<br/>KB sentences, city checks,<br/>what was saved, one question"]
  finalize -.-> leadsdb
  welcome --> finalize
  closing --> finalize
  redirect --> finalize
  finalize --> reply(["reply + sources + steps + answered_by"])
  memory[("Postgres checkpointer<br/>thread = session_id")] -.-> understand
  finalize -.-> memory
```

## Frontend: chat page and sales dashboard (T10, T10c, T11, `frontend/`)
React 19 + Vite 8, plain JavaScript and CSS, lucide-react icons, Plus Jakarta Sans (self-hosted), no router package. Behaviour and design: PLAN.md parts 3.14 (chat) and 3.15 (dashboard); credits: `docs/CREDITS.md`.

| Module | Role |
|---|---|
| `src/main.jsx`, `src/App.jsx`, `src/router.jsx` | The entry (`styles.css` first, so a page's own CSS can adjust it); the router: `/` chat, `/leads`, `/leads/<id>`, `/support`, else "Page not found", and the tab's title; `usePath()` (`useSyncExternalStore`), `navigate()` (History API), `<Link>` (a real link; plain left clicks stay in the app) |
| `src/api.js` | `apiUrl(path)` (VITE_API_URL, trailing slashes removed, default `http://127.0.0.1:8000`), `getWelcome()`, `postChat({message, sessionId})`, `ApiError` (`rate_limit` / `invalid` / `unauthorized` / `not_found` / `server` / `network`, plus the API's `detail`), `errorFrom` (429 → `detail` + `retry_after` or the `Retry-After` header; 422 → first field message; else a generic message); T11: `getLeads`, `getLead`, `getSupportRequests` (the sales key in `X-API-Key`, never in the URL; the answer's shape checked); 90 s timeout; T13a `pingServer()` (one `GET /api/health/live` per page load, never throws); T13d: `postChat` gives each message a `client_message_id` and sends it once more (after `RETRY_DELAY_MS`) when no reply arrived at all, so the server answers with the reply it already sent |
| `src/storage.js` | `loadChat` / `saveChat` / `clearChat`: session id, transcript and lead in sessionStorage; `loadSalesView` / `saveSalesView`: the Sales view switch in localStorage (on by default); T11: `loadSalesKey` / `saveSalesKey` / `clearSalesKey`: the sales key in sessionStorage only; never throws |
| `src/useChat.js` | The conversation hook: messages, lead, session id, draft, `pending` / `slow` (8 s; T13a: `STILL_WORKING`, neutral words), `notice`, `waitLeft` (429 countdown), `send(text)` (one at a time; on failure the text goes back into the draft; each new message gets an `at` time), `newChat()` |
| `src/useWelcome.js` | `GET /api/welcome` once, `retry()` after an error (T13a: no waking note of its own) |
| `src/useServerWake.js`, `src/components/WakeNote.jsx` | T13a: `useServerWake()` is true while the page-load ping still waits after 1.5 s; `WakeNote` (rendered once in `App.jsx`, every page) shows "Waking up the server. The first reply can take up to a minute." in a fixed, polite live region |
| `src/ChatPage.jsx`, `src/useWideScreen.js` | The chat page (was `App.jsx` until T11): the frame grid (header, sidebar, chat card, sales card; no sidebar below 1100 px; phone strip below 900 px), `useSalesView`, welcome + starters, typing indicator, screen-reader announcer; the 900 px media query as a hook |
| `src/components/Header.jsx`, `Sidebar.jsx`, `Logo.jsx` | Brand, Sales view switch (`role="switch"`), New chat; `Sidebar` (`current`, `staff`, `brand`) and `SectionNav`: Chat, and Leads and Support requests only with `staff` (the chat page passes the Sales view switch), the current one a marker, the others links; the rooftop photo (CSS background); the sun-and-horizon mark (`Logo`, and `SunMark` for the assistant avatar) |
| `src/components/ReplyText.jsx` | Reply text as React text nodes: paragraphs, lists (`<ol start>`), "# " headings, bold and italics with stray asterisks dropped (underscores kept), `[KB Section N: Title]` → a quiet "KB N" tag |
| `src/components/SourceChips.jsx`, `StepsPanel.jsx`, `Message.jsx` | Source pills (`<details>`: book icon, "KB 8 · title", snippet on open); "How I answered · N steps · X s" (`<details>`, the bubble's footer: label, detail, time in seconds; T12a: a reply guard that fired shows as "Safety check on the reply"; T13d: a last line from the API's `answered_by` when no model wrote the reply, `answerLine`); one message with its avatar and `<time>` |
| `src/components/LeadCard.jsx` | "Sales view (demo)": name, "City · type", status badge and score (only when sent), saved or not, "Details collected" (`checklist(lead)` from `known_fields`), still needed / would add points, callback; designed empty state. Exports `StatusBadge` (`compact`: no icon) and `DETAILS` for the dashboard |
| `src/components/Composer.jsx` | Textarea (Enter sends, Shift+Enter new line, 1,000 characters), Send / "Wait N s" |
| `src/dashboard/Dashboard.jsx`, `UnlockCard.jsx` | T11: the staff shell (wide sidebar, header with search, Refresh, Lock; tabs below 1280 px) and the unlock gate; Lock and any 401 forget the key; `unlockMessage` (401, 429 with the wait, 503 in the API's words) |
| `src/dashboard/LeadsView.jsx`, `LeadTable.jsx`, `LeadDetail.jsx`, `SupportView.jsx` | Count cards (API `counts`), status chips, the table (desktop) or cards (phone), Load more, the selected lead (T12a order: name, badge and score; full next action, callback, contact; score breakdown and caps, reason, details collected), support requests; loading, empty and error states |
| `src/dashboard/useSalesData.js`, `format.js`, `dashboard.css` | Paged lists (stale answers dropped, 401 → unlock), the lead detail, `useDebounced` (T13a: `useSlow` removed); pure text helpers (one-line requirement, first sentence, "2 h ago", phone format); the dashboard's styles on the shared tokens |
| `frontend/vercel.json` | T13a: every path → `/index.html` on Vercel (deep links); `src/vercel.test.js` |
| `src/*.test.js(x)`, `src/contrast.test.js` | Vitest + Testing Library, `fetch` stubbed (`src/test-utils.js`: health, chat and sales routes, lead fixtures; `test-setup.js` forgets the ping after each test); the contrast of every colour pair in light and dark, read from `styles.css` |

```mermaid
flowchart LR
  router["App.jsx: router"] -->|"/"| page["ChatPage.jsx"]
  router --> wake["WakeNote: one GET /api/health/live per page load<br/>(note while it waits past 1.5 s)"]
  wake --> api
  router -->|"/leads, /leads/id, /support"| dash["dashboard/Dashboard.jsx"]
  dash -->|"no key"| unlock["UnlockCard: GET /api/leads?limit=1"]
  dash --> leads["LeadsView: counts, chips, table, detail"]
  dash --> support["SupportView"]
  leads -->|"GET /api/leads?status, q, sort, offset + X-API-Key"| api
  leads -->|"GET /api/leads/id"| api
  support -->|"GET /api/support-requests"| api
  keystore[("sessionStorage: sales key")] <--> dash
  page --> welcome["useWelcome"]
  page --> chat["useChat"]
  welcome -->|"GET /api/welcome"| api["api.js: apiUrl, ApiError, pingServer"]
  chat -->|"POST /api/chat {message, session_id}"| api
  api --> backend["FastAPI (Render, or local 127.0.0.1:8000)"]
  chat <--> store[("sessionStorage: session id, transcript, lead")]
  chat --> view["Message: ReplyText, SourceChips, StepsPanel"]
  chat --> card["LeadCard (Sales view, demo): identity, score, checklist, callback"]
  prefs[("localStorage: Sales view switch")] --> card
  chat --> composer["Composer + notices (429 countdown, 422, errors)"]
```

# ARCHITECTURE.md

Short map of how the code is built. Updated at the end of each task. See `docs/PLAN.md` for the full design.

## Backend modules (as of T10)
| Module | Role |
|---|---|
| `app/main.py` | FastAPI app, startup hook (`init_db`), request-id + logging middleware (skips `QUIET_PATHS`: `/api/health/live`; adds `ip_source` for rate-limited requests, never the IP), error handlers (`ApiError`, 422 with one message per field, database down → 503), CORS (GET/POST; `Content-Type`, `X-API-Key`, `X-Request-ID`; exposes `Retry-After`), routers |
| `app/core/config.py` | `Settings` (pydantic-settings, reads `backend/.env`), `get_settings()` |
| `app/core/rate_limit.py` | `RateLimiter(limits)`: in-memory fixed windows per key, thread-safe; `hit()` counts or returns the seconds to wait (a refused request is not counted), `blocked()` only checks |
| `app/api/security.py` | `client_ip` (the `CLIENT_IP_HEADER` value if it is one valid IP, else the connection address; never `X-Forwarded-For`), `limit_key` (IPv6 per /64), `chat_rate_limit` (10/min + 100/day per IP → 429 + Retry-After), `require_sales_key` (`X-API-Key`, constant-time, fails closed, 10 wrong keys per 15 min per IP → 429), `ApiError` + `error_response` (the JSON error shape) |
| `app/api/sales.py` | Sales router (key required): `GET /api/leads`, `GET /api/leads/{id}`, `POST /api/leads`, `POST /api/callback`, `GET /api/support-requests`, all through `app/leads/service.py`; never returns `session_id` |
| `app/core/logging.py` | JSON log formatter, `request_id_var` |
| `app/core/db.py` | `Base`, `get_engine()` (Neon, pre-ping), `check_db()`, `init_db(engine)` (pgvector + `create_all`) |
| `app/core/llm.py` | `embedder()` and `query_embedder()` (google-genai, one Content per text, batches of 20, LRU cache for queries). `GeminiChat`: adds our `http_options` (15 s, 1 retry, none on 429) to every Gemini request. `chat_llm(model, tools=, schema=)`: Gemini with a Groq fallback, built separately for plain, tool-bound and structured-output use (Groq: `parallel_tool_calls=False`). `main_llm()`, `fast_llm()`, `invoke_llm()`, `provider_of()`, `reply_text()` |
| `app/core/prompts.py` | `load_prompt(name)`: reads `prompts/<name>.md` (cached) |
| `app/agent/state.py` | `AgentState` (persisted `messages` and `lead`; per-turn fields) and `turn_input()`, which resets the per-turn fields |
| `app/agent/understand.py` | `Understanding` schema (intents, `search_query`, details generated from `LeadData`), `load_kb_hints` (section titles + table summary chunks from `kb_chunks`), `understand()` |
| `app/agent/tools.py` | `make_tools`: `search_knowledge_base`, `check_service_area` (content + KB chunk artifacts) and, with `LeadOps`, `create_lead`, `update_lead`, `schedule_callback`, `create_support_request` (no arguments; read the draft via `InjectedState`; no score in results); `tool_step` |
| `app/agent/graph.py` | `build_graph(AgentDeps)`: understand → retrieve → agent ⇄ tools → finalize, plus welcome / closing / redirect; `build_agent()`, `get_agent()` (FastAPI dependency) |
| `app/agent/leads.py` | `LeadOps` (leads database, reason model, `reason_now`), the lead progress block (`status_block`), `confirmed_city`, `pending_changes` (write-once sync), `lead_summary` for the API |
| `app/core/tracing.py` | `configure_tracing`: optional LangSmith; copies LANGSMITH_* settings into os.environ only when tracing is on and a key is set |
| `app/agent/memory.py` | `get_checkpointer()` (PostgresSaver over a psycopg pool), `setup_checkpointer()` (never fails the boot) |
| `app/agent/welcome.py` | Fixed texts: welcome (F0), starter questions, closing line, off-topic redirect |
| `app/api/welcome.py` | `GET /api/welcome` |
| `app/api/health.py` | `GET /api/health` (status, db, kb_chunks); `GET /api/health/live` (`{"status":"ok"}`, no DB, Render's health check) |
| `app/api/chat.py` | `POST /api/chat`: rate limited (`chat_rate_limit`); `session_id` must be a UUID v4 (the server issues one when none is sent); validates the message, runs the agent with `thread_id` = `session_id`, returns reply, sources, steps, `lead` (summary; status and score only with `CHAT_SHOW_LEAD_SCORE`) and degraded. One turn per session (striped locks, 60 s timeout); a pending lead reason is written after the reply (background task). DB errors → 503; anything else → 200 degraded |
| `app/rag/answer.py` | Helpers the agent reuses: `format_context` (`[KB Section N: Title]` labels), `rag_message`, `cited_sections`, `pick_sources` (one per cited section), `normalize_citations`, `retrieve_step`, `LLM_DOWN_REPLY` |
| `app/rag/pdf_parser.py` | `parse_kb(path)`: KB PDF to numbered sections with text lines and tables (pdfplumber; cover, header and footer removed) |
| `app/rag/chunker.py` | `Chunk`, `chunk_sections()`: one chunk per table row (KB Sections 8-12) plus one `table_summary` chunk listing every row (8, 10), one per FAQ item (15), prose split at ~1,200 chars, Sections 16-19 tagged `policy`; `embed_text()` = `title: ... \| text: ...` |
| `app/rag/models.py` | `KbChunk` (`kb_chunks`: vector + generated tsvector with GIN index) and `ServiceArea` (`service_areas`, unique on city + region) |
| `app/rag/ingest.py` | `python -m app.rag.ingest`: parse, chunk, embed all first, then replace all rows in ONE transaction; `--dry-run` prints chunks and touches no DB or key |
| `app/rag/retriever.py` | `search(query, k, mode)`: keyword (stemmed words OR'd, words in more than 30% of chunks dropped, `ts_rank_cd`) + vector (cosine), RRF fusion (k = 60, keyword weight 0.5), relevance gate (similarity only when a vector search ran), keyword fallback with `degraded`; policy chunks never returned |
| `app/leads/schemas.py` | `LeadData` (the lead contract: every field optional, null = unknown, units in names, per-field `description` for T8's structured output, `from_raw` keeps valid values and reports invalid ones), `LeadView` and the result models |
| `app/leads/validators.py` | Pure validators: Indian mobile → `+91XXXXXXXXXX`, email syntax (email-validator, no network), ₹ amounts (`₹4,000`, `4k`, `1.2 lakh`), kW, kWh, names, text; plain error messages |
| `app/leads/scoring.py` | `score_lead` (rubric, caps at WARM and 74, breakdown JSON), `status_for`, `recommended_action`, `missing_for_creation` |
| `app/leads/reason.py` | `write_reason`: one sentence from `fast_llm` (breakdown only, no personal details or money), template fallback |
| `app/leads/models.py` | `Lead` (`leads`), `Callback` (`callbacks`, status `requested`), `SupportRequest` (`support_requests`) |
| `app/leads/service.py` | `create_lead` (refuses until the minimum fields, one lead per session), `update_lead` (merge, re-locate, re-score), `get_lead`, `list_leads`, `schedule_callback`, `create_support_request`, `list_support_requests`; logs ids, status and score only |
| `app/services/service_areas.py` | `ServiceAreas.lookup(city, state)`: alias map + rapidfuzz (score 85); loads from the `service_areas` table (`get_service_areas()`, cached) |
| `scripts/list_models.py` | Lists Gemini chat and embedding models for the key |
| `scripts/search_kb.py` | Prints the mode, `no_context` and the top chunks with scores and ranks |
| `scripts/serve_test_branch.py` | Dev only: runs the API on 127.0.0.1:8000 with the chat agent's lead writes and the sales API (`leads_engine`) on `TEST_DATABASE_URL`; KB and service areas read from main; chat memory in the process (`InMemorySaver`). `engine_for_test_branch` refuses an empty or same-as-main URL |
| `scripts/eval_retrieval.py` | Embeds the 24 KB and 8 off-topic questions once, runs `search()` in all three modes (k = 10), prints hit@1, hit@3, MRR, the misses and the gate results, and writes `docs/EVAL.md` (its "Tuning notes" section is hand-written and kept). `--rewrite` adds the understand step's rewritten query (`rewrite` and `fused` modes) and the small-to-big report |

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

## Chat v0 (T5)
```mermaid
flowchart LR
  msg["message: control chars stripped, 1-1,000 chars"] --> search["search: hybrid, top 4"]
  search -->|"no_context"| fixed["fixed reply, no LLM call"]
  search --> prompt["system_prompt.md + rag_prompt.md with [KB Section N: Title] blocks"]
  prompt --> gemini["Gemini main model: thinking minimal, 15 s, 1 retry, none on 429"]
  gemini -.->|"any error"| groq["Groq fallback: 15 s, no retry"]
  gemini --> reply["reply text only, citations → sources (one per section), steps"]
  groq --> reply
  groq -.->|"fails too"| degraded["200, degraded reply"]
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
  q --> emb["embed_query (10 s, 1 retry)"]
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
Files at the repo root: `Dockerfile` (python:3.12-slim, only the app package and `prompts/`, non-root user, one uvicorn worker on `$PORT`), `.dockerignore`, `render.yaml` (one free web service, region Singapore, next to Neon). Steps and limits: `docs/DEPLOY.md`.
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
  score --> reason["write_reason: fast_llm or template"]
  score --> action["recommended_action"]
  reason --> db[("leads, callbacks, support_requests")]
  action --> db
```

## Agent graph (T8a)
One customer message; state checkpointed in Postgres per `session_id`. Details: PLAN.md part 3.1.
```mermaid
flowchart TD
  msg(["POST /api/chat"]) --> understand["understand<br/>fast LLM, structured output:<br/>intents, search_query, lead details"]
  understand -->|"only hello / what can you do"| welcome["welcome text"]
  understand -->|"only thanks / bye"| closing["closing line"]
  understand -->|"only off-topic, no query"| redirect["polite redirect"]
  understand -->|"search_query, not details-only"| retrieve["retrieve<br/>hybrid search, top 4"]
  understand -->|"anything else"| agent
  retrieve --> agent["agent<br/>main LLM + tools<br/>(Gemini, Groq fallback)"]
  agent -->|"tool calls (max 4 rounds)"| tools["tools<br/>search_knowledge_base, check_service_area<br/>create_lead, update_lead,<br/>schedule_callback, create_support_request"]
  tools -.->|"lead tools (test branch in live tests)"| leadsdb[("leads, callbacks,<br/>support_requests")]
  tools --> agent
  agent -->|"answer"| finalize["finalize<br/>reply, sources, steps,<br/>write-once lead sync, lead summary"]
  finalize -.-> leadsdb
  welcome --> finalize
  closing --> finalize
  redirect --> finalize
  finalize --> reply(["reply + sources + steps"])
  memory[("Postgres checkpointer<br/>thread = session_id")] -.-> understand
  finalize -.-> memory
```

## Chat page (T10, `frontend/`)
React 19 + Vite 8, plain JavaScript and CSS. Behaviour: PLAN.md part 3.14.

| Module | Role |
|---|---|
| `src/api.js` | `apiUrl(path)` (VITE_API_URL, trailing slashes removed, default `http://127.0.0.1:8000`), `getWelcome()`, `postChat({message, sessionId})`, `ApiError` (`rate_limit` / `invalid` / `server` / `network`), `errorFrom` (429 → `detail` + `retry_after` or the `Retry-After` header; 422 → first field message; else a generic message); 90 s timeout |
| `src/storage.js` | `loadChat` / `saveChat` / `clearChat`: session id, transcript and lead in sessionStorage; never throws |
| `src/useChat.js` | The conversation hook: messages, lead, session id, draft, `pending` / `slow` (8 s), `notice`, `waitLeft` (429 countdown), `send(text)` (one at a time; on failure the text goes back into the draft), `newChat()` |
| `src/useWelcome.js` | `GET /api/welcome` once; `slow` after 3 s (the waking-up notice), `retry()` after an error |
| `src/App.jsx` | Layout (header, transcript, notices, composer, sales panel or phone strip at 900 px), welcome + starters, typing indicator, screen-reader announcer |
| `src/components/ReplyText.jsx` | Reply text as React text nodes: paragraphs, lists, bold, `[KB Section N: Title]` → "KB Section N" tag |
| `src/components/SourceChips.jsx`, `StepsPanel.jsx`, `Message.jsx` | Source chips (`<details>`: section + title, snippet on open); "How I answered" (`<details>`: step label, detail, time); one message |
| `src/components/LeadCard.jsx` | "Sales view (demo)": saved or not, still needed / would add points, callback; status and score only when sent |
| `src/components/Composer.jsx` | Textarea (Enter sends, Shift+Enter new line, 1,000 characters), Send / "Wait N s" |
| `src/*.test.js(x)` | Vitest + Testing Library, `fetch` stubbed (`src/test-utils.js`) |

```mermaid
flowchart LR
  page["App.jsx"] --> welcome["useWelcome"]
  page --> chat["useChat"]
  welcome -->|"GET /api/welcome (3 s: waking-up notice)"| api["api.js: apiUrl, ApiError"]
  chat -->|"POST /api/chat {message, session_id}"| api
  api --> backend["FastAPI (Render, or local 127.0.0.1:8000)"]
  chat <--> store[("sessionStorage: session id, transcript, lead")]
  chat --> view["Message: ReplyText, SourceChips, StepsPanel"]
  chat --> card["LeadCard (Sales view, demo)"]
  chat --> composer["Composer + notices (429 countdown, 422, errors)"]
```

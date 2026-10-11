# ARCHITECTURE.md

Short map of how the code is built. Updated at the end of each task. See `docs/PLAN.md` for the full design.

## Backend modules (as of T8a)
| Module | Role |
|---|---|
| `app/main.py` | FastAPI app, startup hook (`init_db`), request-id + logging middleware (skips `QUIET_PATHS`: `/api/health/live`), CORS, routers |
| `app/core/config.py` | `Settings` (pydantic-settings, reads `backend/.env`), `get_settings()` |
| `app/core/logging.py` | JSON log formatter, `request_id_var` |
| `app/core/db.py` | `Base`, `get_engine()` (Neon, pre-ping), `check_db()`, `init_db(engine)` (pgvector + `create_all`) |
| `app/core/llm.py` | `embedder()` and `query_embedder()` (google-genai, one Content per text, batches of 20, LRU cache for queries). `GeminiChat`: adds our `http_options` (15 s, 1 retry, none on 429) to every Gemini request. `chat_llm(model, tools=, schema=)`: Gemini with a Groq fallback, built separately for plain, tool-bound and structured-output use (Groq: `parallel_tool_calls=False`). `main_llm()`, `fast_llm()`, `invoke_llm()`, `provider_of()`, `reply_text()` |
| `app/core/prompts.py` | `load_prompt(name)`: reads `prompts/<name>.md` (cached) |
| `app/agent/state.py` | `AgentState` (persisted `messages` and `lead`; per-turn fields) and `turn_input()`, which resets the per-turn fields |
| `app/agent/understand.py` | `Understanding` schema (intents, `search_query`, details generated from `LeadData`), `load_kb_hints` (section titles + table summary chunks from `kb_chunks`), `understand()` |
| `app/agent/tools.py` | `make_tools`: `search_knowledge_base`, `check_service_area` (content + KB chunk artifacts); `tool_step` |
| `app/agent/graph.py` | `build_graph(AgentDeps)`: understand → retrieve → agent ⇄ tools → finalize, plus welcome / closing / redirect; `build_agent()`, `get_agent()` (FastAPI dependency) |
| `app/agent/memory.py` | `get_checkpointer()` (PostgresSaver over a psycopg pool), `setup_checkpointer()` (never fails the boot) |
| `app/agent/welcome.py` | Fixed texts: welcome (F0), starter questions, closing line, off-topic redirect |
| `app/api/welcome.py` | `GET /api/welcome` |
| `app/api/health.py` | `GET /api/health` (status, db, kb_chunks); `GET /api/health/live` (`{"status":"ok"}`, no DB, Render's health check) |
| `app/api/chat.py` | `POST /api/chat`: validates the message, runs the agent with `thread_id` = `session_id`, returns reply, sources, steps, `lead` (null until T8b) and degraded. DB errors → 503; anything else → 200 degraded |
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
| `app/leads/service.py` | `create_lead` (refuses until the minimum fields, one lead per session), `update_lead` (merge, re-locate, re-score), `get_lead`, `list_leads`, `schedule_callback`, `create_support_request`; logs ids, status and score only |
| `app/services/service_areas.py` | `ServiceAreas.lookup(city, state)`: alias map + rapidfuzz (score 85); loads from the `service_areas` table (`get_service_areas()`, cached) |
| `scripts/list_models.py` | Lists Gemini chat and embedding models for the key |
| `scripts/search_kb.py` | Prints the mode, `no_context` and the top chunks with scores and ranks |
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
  mw --> chat["POST /api/chat"]
  chat --> agent["agent graph (see Agent graph)"]
  agent --> neon
  agent -.->|"DB error"| e503["503, friendly message"]
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
Called by T8's tools and T9's API (no endpoint yet). Rules in plain words: `docs/SCORING.md`.
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
  understand -->|"search_query"| retrieve["retrieve<br/>hybrid search, top 4"]
  understand -->|"anything else"| agent
  retrieve --> agent["agent<br/>main LLM + tools<br/>(Gemini, Groq fallback)"]
  agent -->|"tool calls (max 4 rounds)"| tools["tools<br/>search_knowledge_base<br/>check_service_area"]
  tools --> agent
  agent -->|"answer"| finalize["finalize<br/>reply, sources, steps"]
  welcome --> finalize
  closing --> finalize
  redirect --> finalize
  finalize --> reply(["reply + sources + steps"])
  memory[("Postgres checkpointer<br/>thread = session_id")] -.-> understand
  finalize -.-> memory
```

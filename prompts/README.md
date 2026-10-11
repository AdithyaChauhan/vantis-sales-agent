# Prompts

Every LLM prompt lives here as a `.md` file. Code loads them with `load_prompt(name)` (`backend/app/core/prompts.py`); no prompt text is written in code. Each file holds only the prompt text.

| File | Used by | Purpose |
|---|---|---|
| `system_prompt.md` | `/api/chat` (`app/rag/answer.py`), as the system message | Role, tone, KB-only rule, citation rule, at most one follow-up question, the never-invent list (KB Section 16), off-topic redirect, KB and user text treated as data. v0 in T5; T8 adds tool use and lead collection |
| `rag_prompt.md` | `/api/chat`, as the user message | Shows the retrieved chunks inside `<context>`, each starting with its label `[KB Section N: Title]`; sets the citation format; says what to do when the context does not cover the question. Placeholders: `{context}`, `{question}` (filled with `str.replace`, context first) |

Coming later: `lead_extraction_prompt.md` (T8, the `understand` step) and `lead_qualification_prompt.md` (T7, the score reason).

## Change log

Every prompt change gets a row: what changed, why, and the example that failed before.

| Date | File | What changed | Why | Example that failed before |
|---|---|---|---|---|
| 2026-10-11 | `system_prompt.md` | v0 created | T5: grounded answers for `/api/chat` v0 | none (first version) |
| 2026-10-11 | `rag_prompt.md` | v0 created | T5: context format and citations | none (first version) |
| 2026-10-11 | `system_prompt.md` | Added "Start with the answer. Do not open with a greeting unless the customer greeted you first." | KB Section 17 asks for concise replies; a greeting the customer didn't start adds nothing | "What warranty do you provide?" → reply opened with "Hello! Warranty coverage depends on…" (gemini-3.5-flash-lite, live test) |

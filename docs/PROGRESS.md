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
- Decisions: stack per PLAN §3; Gemini main + Groq fallback; Neon; Render + Vercel; mock WhatsApp skipped; KB PDF stays out of git (README says where to put it)
- Open issues: check the submission form fields · check free-tier limits in the AI Studio and Groq consoles
- Next: T1 Foundation

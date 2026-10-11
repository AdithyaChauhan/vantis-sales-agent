---
name: handoff
description: End-of-task wrap-up for the Vantis project. Runs the tests, updates docs/PROGRESS.md (and docs/ARCHITECTURE.md when the structure changed), commits, and prints the progress entry for Adithya to paste to his reviewer.
disable-model-invocation: true
allowed-tools: Bash(git add *) Bash(git commit *) Bash(git status *) Bash(git diff *)
---

Wrap up the task you just finished. Do the steps in order and stop at the first failure.

1. **Tests.** If `backend/` exists, run `cd backend && python -m pytest -q`. If anything fails, stop: report the failures and do not commit.
2. **Frontend.** If files under `frontend/` changed in this task, run `cd frontend && npm run build`. If the build fails, stop and do not commit.
3. **Progress log.** Append an entry to `docs/PROGRESS.md` using the template at the top of that file. 12 lines max. List only what you actually ran and saw.
4. **Architecture doc.** If you added or renamed modules, endpoints, tables, graph nodes or tools, update `docs/ARCHITECTURE.md` (create it if missing). Keep it short, with its Mermaid diagrams matching the code.
5. **Prompt log.** If any file in `prompts/` changed, make sure `prompts/README.md` has a log line for it: what changed, why, and the example that failed before.
6. **Commands.** If you added a command people need to run, add it to the Commands section of `CLAUDE.md`.
7. **Safety check.** Run `git status`. Make sure no `.env` file, PDF, key or token is about to be committed.
8. **Commit.** Stage this task's files and commit with a conventional message, for example `feat(rag): hybrid retriever with RRF`. The commit message must have a blank line after the subject line. Do not push.
9. **Report.** Print the new PROGRESS.md entry in a code block, then stop.

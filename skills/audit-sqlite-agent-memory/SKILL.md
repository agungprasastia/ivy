---
name: audit-sqlite-agent-memory
description: Inspect an AI agent's SQLite memory database to determine what the agent actually persists across sessions (conversation history vs. long-term user profile). Use when asked "do you remember me across sessions?", when debugging memory/persistence bugs, or when verifying that profile facts, preferences, and cron state are truly stored rather than inferred from the current chat context.
status: draft
---
# Audit an Agent's SQLite Memory Store

Goal: replace speculation ("I think I remember you") with hard evidence — table list, row counts, schema source, and sample rows — so you can state exactly what persists and what does not.

## When to use

- User asks whether the agent remembers them / their preferences across sessions.
- Debugging why a "memory" feature appears to work in-session but not in a new session.
- Verifying cron jobs, user profile facts, or FTS search indexes are populated.

## Procedure

1. **Locate the database and its owning code.**
   - Find the DB file (commonly `ivy.db`, `memory.db`, or similar) and the module that opens it (e.g. `ivy/memory/store.py`).
   - Grep the source for `CREATE TABLE`, `set_preference`, `get_user_profile`, `INSERT INTO` to learn intended semantics before touching data.

2. **Never assume a `sqlite3` CLI exists.** On Windows it is often absent, and inline one-liners/heredocs are error-prone. Instead write a tiny throwaway Python probe file, run it, then delete it.

   ```python
   import sqlite3
   conn = sqlite3.connect(r"<ABS_PATH_TO_DB>")
   rows = conn.execute(
       "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
   ).fetchall()
   print("tables:", [r[0] for r in rows])
   for (t,) in rows:
       try:
           n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
           print(f"{t}: {n}")
       except Exception as e:
           print(f"{t}: error {e}")
   ```

3. **Interpret the table inventory deliberately.**
   - A `messages` (or `messages_fts*`) family = per-session transcript. `messages_fts`, `_data`, `_idx`, `_content`, `_docsize`, `_config` are FTS5 shadow tables — they duplicate message text, so **do not sum them as "extra memories."** Count messages once.
   - `user_profile` = cross-session preferences/facts. **If it has 0 rows, the agent has no durable memory of the user**, regardless of how conversational it sounds.
   - `cron_jobs` / `cron_runs` = scheduled-task state; 0 rows means no automations registered.
   - `sessions` row count tells you how many distinct conversation threads exist (often just one, e.g. `cli:main`).

4. **Sample actual content** with a second small probe: select `id, role, substr(content,1,160)` from messages ordered by id, and separately dump `user_profile` rows. Truncate long text so output stays readable.

5. **Report the distinction explicitly.** Separate:
   - *Durable memory*: rows in the profile table (possibly zero).
   - *Session memory*: rows in `messages` for the current session only.
   - *Nothing*: everything the model merely "sounds like" it remembers from the in-context transcript.

6. **Clean up** the temporary probe script when done.

## Pitfalls

- Do not run destructive or broad shell commands (mass delete, registry edits, recursive operations) just to inspect a DB; safety approval policies will reject them and you waste turns. Read-only `SELECT` queries are enough.
- Use absolute paths with raw strings (`r"C:\..."`) on Windows; the working directory may be under `AppData\Roaming\...` while the project lives on another drive such as `D:\...`.
- Store temp scripts in the approved temp dir (e.g. `C:\Users\<user>\AppData\Local\Temp\opencode`) rather than scattering files in the repo.
- An empty `assistant` content row or a lone `clear` command in the transcript is expected noise, not corruption.
- Do not claim persistence based on the conversation alone — always cite row counts.

## Success criteria

You can answer with: "`user_profile` has N rows; `messages` has M rows in session S; therefore durable memory is X and session memory is Y."

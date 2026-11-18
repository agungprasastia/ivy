# AGENTS.md

Guidelines for coding agents working in the **Ivy** repository.

## Overview
Ivy is a cross-platform Python 3.11+ AI coding and productivity agent with a Rich + `prompt_toolkit` TUI, SQLite + FTS5 session persistence, skills engine, cron scheduler, and Telegram/Discord gateway.

## Architecture Layers
- `ivy/config.py`: Pydantic settings loaded from `.env` (`IvyConfig`, `load_config`, `update_env_file`).
- `ivy/core/`: Agent runtime loop (`agent.py`), OpenAI/Anthropic/LiteLLM-compatible client (`llm.py`), system prompt & secret redaction/head+tail tool formatting (`prompt.py`), isolated subagent runner (`subagent.py`), shared dataclasses/types (`types.py`).
- `ivy/tools/`: `ToolRegistry` (`registry.py`), workspace file I/O (`files.py`), guarded shell execution (`terminal.py`), background process registry & `process` tool (`process_registry.py`), interactive safety policy (`approval.py`), reversible file snapshots (`checkpoint.py`), Python code sandbox (`runner.py`), web search/fetch (`web.py`), todo list (`todo.py`), session search (`session_search.py`), user preferences (`profile.py`), clarifying prompts (`clarify.py`), cron tool (`cron_tool.py`), subagent delegation (`delegate.py`), and stdio MCP bridge (`mcp.py`).
- `ivy/skills/`: Markdown `SKILL.md` loader/matcher (`engine.py`), reader tool (`tool.py`), runtime skill creator/patcher (`skill_manager.py`), and LLM skill synthesizer (`generator.py`).
- `ivy/memory/`: SQLite WAL + FTS5 persistence (`store.py`) and token-budget context compression (`context.py`).
- `ivy/cron/`: `croniter`-based background scheduler (`scheduler.py`) and task models (`task.py`).
- `ivy/gateway/`: Optional Telegram/Discord bot adapters (`server.py`, `handlers.py`, `router.py`, `formatters.py`).
- `ivy/tui/`: REPL entrypoint & loop (`app.py`), slash commands & pickers (`commands.py`), prompt/toolbar/completions (`prompt.py`), fuzzy arrow-key picker (`picker.py`), Rich renderers (`views.py`), themes (`theme.py`), `@` expansion & redacting exporters (`helpers.py`), diagnostics (`doctor.py`).

## Built-in Tools & MCP
16 built-in tools: `read_file`, `write_file`, `patch_file`, `list_directory`, `search_files`, `terminal` (foreground + `background=true`), `process`, `code_runner`, `web_search`, `web_fetch`, `read_skill`, `skill_manage`, `session_search`, `todo_list`, `remember_preference`, `clarify`, plus conditional `cron` and `delegate_task`, and optional stdio MCP tools loaded from `.mcp.json` via `MCPManager`.

## Running Tests & Diagnostics (Windows PowerShell)
- Run full test suite (do not use `bash.exe` on Windows):
  ```powershell
  & ".\.venv\Scripts\pytest.exe" -p no:cacheprovider -v
  ```
- Inspect repository diagnostics:
  ```powershell
  & ".\.venv\Scripts\ivy.exe" doctor
  ```

## Coding & Safety Conventions
- **No unnecessary runtime deps:** Keep `pyproject.toml` dependencies minimal (standard library + existing 7 runtime packages).
- **Monkeypatch stability:** Tests patch symbols imported onto `ivy.tui.app` (`copy_to_clipboard`, `run_interactive_picker`, `run_git_diff`, `create_prompt_session`, `run_doctor_checks`, `render_doctor_report`). Keep those imports exposed in `ivy/tui/app.py` and resolve them through `_app_ns()` in `ivy/tui/commands.py`.
- **Context & secret hygiene:** Bound tool results with `truncate_head_tail` (16,000 chars default) and scrub secrets via `redact_secrets` before persisting/exporting or feeding tool outputs back to the model.
- **Reversibility & safety:** Snapshot file mutations with `FileCheckpointStore` before writing/patching and route terminal commands through `SafetyApprovalPolicy`.

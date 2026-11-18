<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/ivy-mascot-dark.png">
    <source media="(prefers-color-scheme: light)" srcset=".github/assets/ivy-mascot.png">
    <img src=".github/assets/ivy-mascot.png" width="230" alt="Ivy — Autonomous AI Agent">
  </picture>
</p>

<p align="center">
  Autonomous, self-improving Python agent. Built cleanly for Python 3.11+ and managed with Astral <code>uv</code>.
</p>

<p align="center">
  <a href="https://github.com/agungprasastia/ivy/actions/workflows/ci.yml"><img src="https://github.com/agungprasastia/ivy/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-10B981.svg" alt="License: Apache-2.0" /></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/managed%20with-uv-DE5FE9.svg?logo=uv&logoColor=white" alt="Managed with uv" /></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/code%20style-ruff-D7FF64.svg?logo=ruff&logoColor=111827" alt="Code style: ruff" /></a>
  <a href="https://github.com/agungprasastia/ivy/stargazers"><img src="https://img.shields.io/github/stars/agungprasastia/ivy?style=flat&color=10B981" alt="GitHub Stars" /></a>
</p>

---

## Overview

Ivy is a modular, async-first AI agent ecosystem with a terminal-first UX:

| Capability | Description |
|---|---|
| **ReAct Execution Engine** | Iterative reasoning → tool execution → observation, with automatic error and safety-rejection recovery. |
| **Universal LLM Adapter** | OpenAI, Anthropic, Gemini, OpenRouter, Ollama and vLLM routed through the native `openai` / `anthropic` async SDKs (prebuilt wheels, no C compiler needed). |
| **Tool Runtime + Safety Guard** | PowerShell/Bash terminal, file tools, sandboxed Python `code_runner`, web fetch/search — with destructive-command interception (`SafetyApprovalPolicy`). |
| **Closed Learning Loop** | Dynamic `SKILL.md` discovery plus post-task reflection that writes reusable skills (`SkillGenerator`). |
| **SQLite FTS5 Memory** | Cross-session full-text recall (`messages_fts`), persistent user profile, and sliding context-window compression. |
| **Messaging Gateway + Cron** | Fail-closed `ALLOWED_USERS` whitelist, per-session `TurnLeaseManager`, Telegram (`aiogram`) & Discord (`discord.py`) adapters, `croniter` scheduling. |
| **Rich TUI** | `prompt_toolkit` REPL with slash-command autocomplete, live markdown streaming, interactive pickers, and three themes. |

---

## Installation

Requires Python 3.11+ and Astral `uv`.

```bash
git clone <repo-url> && cd Agent

# Core + CLI/TUI
uv sync

# Including Telegram & Discord gateway adapters
uv sync --extra gateway
```

---

## Configuration

```bash
cp .env.example .env
```

Then edit `.env`. All settings are optional except a provider API key for the model you intend to use.

### Core

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_MODEL` | `gpt-4o-mini` | Active model (see [Model Strings](#model-strings)) |
| `IVY_DB_PATH` | `./ivy.db` | SQLite database for memory & sessions |
| `SKILLS_DIR` | `skills` | Directory scanned for `SKILL.md` |
| `LOG_LEVEL` | `INFO` | Root log level |
| `VERBOSE` | `false` | Debug logging (incl. HTTP SDK logs) |
| `IVY_THEME` | `ivy` | TUI theme: `ivy`, `hermes`, `ocean` |

### Providers

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI (also used for vLLM and custom endpoints) |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `OPENROUTER_API_KEY` | OpenRouter |
| `IVY_API_BASE` / `OPENAI_API_BASE` / `LITELLM_API_BASE` | Custom OpenAI-compatible base URL (e.g. `http://localhost:20128/v1`) |

Ollama and vLLM need no API key.

### Gateway & Safety

| Variable | Purpose |
|---|---|
| `ALLOWED_USERS` | Comma-separated user IDs authorised to drive the bots. **Empty = deny all (fail-closed).** |
| `TELEGRAM_TOKEN` | Telegram bot token (`aiogram`) |
| `DISCORD_TOKEN` | Discord bot token (`discord.py`) |
| `AUTO_APPROVE_SAFETY` | `true` to auto-approve destructive commands at startup (`/yolo` toggles per session) |
| `IVY_RESTRICT_WORKSPACE` | `true` to confine file tools strictly to the current working directory |
| `IVY_ALLOW_PRIVATE_URLS` | `true` to allow `web_fetch` on localhost/private LAN URLs (cloud metadata remains blocked) |

### Model Strings

Provider is inferred from the prefix, so plain names still work:

```
gpt-4o-mini
anthropic/claude-sonnet-4-5
gemini/gemini-2.5-flash
openrouter/meta-llama/llama-3.3-70b-instruct
ollama/qwen2.5
vllm/my-local-model
```

No prefix → OpenAI. A `claude*` name → Anthropic; a `gemini*` name → Gemini.

---

## Usage

### Interactive TUI

```bash
uv run ivy
```

Override the model, resume a named session, or pick a theme:

```bash
uv run ivy --model gemini/gemini-2.5-flash --session my-project
uv run ivy -c                  # resume the most recently updated session (--continue)
uv run ivy --theme ocean
uv run ivy doctor              # run environment, DB, config & skill diagnostics
```

### One-Shot (non-interactive)

```bash
uv run ivy -p "Summarise the open TODOs in src/"
```

### Gateway Daemon

```bash
uv run ivy gateway start     # messaging + cron daemon
uv run ivy gateway status    # show whitelist size
```

### CLI Subcommands (Skills, Cron, Diagnostics)

```bash
uv run ivy doctor
uv run ivy skills list
uv run ivy skills inspect coding-assistant
uv run ivy skills remove <name>
uv run ivy cron list
uv run ivy model              # print the active model
```

---

## TUI Commands

Type `/` for autocomplete, or press **`Ctrl+P`** for the fuzzy command palette, **`Ctrl+G`** to compose in `$EDITOR`, and **`Ctrl+S`** to stash/restore drafts. Prefixes and aliases resolve automatically.

| Command | Action |
|---|---|
| `/setup` | Configure API base URL, API key and model (writes to `.env`) |
| `/model` | Interactive model picker, or `/model <name>` |
| `/session` · `/resume` | Interactive session picker; `/session list`, `/session <id\|#N>` |
| `/new` · `/fork` · `/title` | New session · branch current session · set/view title |
| `/undo` · `/rewind` · `/retry` | Roll back last turn · rewind to any prompt · roll back and re-send |
| `/rollback` | Inspect, diff, or restore automatic file edit checkpoints |
| `/btw` | Ask an off-record side question without mutating session history |
| `/history` · `/save` · `/copy` | Show turns · export secret-redacted `md`/`json` · copy last reply |
| `/status` · `/context` · `/usage` | Session/approval state · token breakdown · API call totals |
| `/compact` | LLM-summarise older turns to reclaim context |
| `/diff` · `/review` | `git diff` in workspace · LLM diff audit (`/review security`) |
| `/init` · `/goal` | Generate `AGENTS.md` · autonomous loop with verification `--gate` |
| `/todo` · `/profile` · `/mcp` · `/ps` | Track tasks · preferences · MCP servers · background processes |
| `/verbose` · `/focus` · `/reasoning` | Cycle verbosity · toggle tool hiding · toggle thinking blocks |
| `/yolo` | Toggle auto-approval of destructive commands for this session |
| `/prompt` | Compose a multiline prompt in `$EDITOR` |
| `/skills` · `/cron` · `/memory` | List/inspect/remove/reflect · manage tasks · FTS5 search |
| `/theme` · `/clear` · `/help` · `/exit` | Interactive theme picker · clear session · help · quit |

### Shell Escape

Prefix a line with `!` to run it locally with **zero LLM tokens** (still subject to the safety guard):

```
!git status --short
```

### Attachments

Mention workspace context inline; Ivy inlines it into the prompt before sending:

```
Explain the failure in @src/ivy/core/agent.py
Review @diff            # current git diff
Review @staged          # staged git diff
Check @url:https://example.com
```

Files are capped at ~12 KB each to protect the context window, tool outputs use head+tail truncation (16 KB max), and known secrets (`sk-...`, `Bearer ...`, API keys) are redacted automatically from tool outputs and `/save` exports.

### Safety Approvals

When a destructive command is intercepted you choose:

```
[y] allow once   [s] allow this command this session   [a] allow all
[n] deny         [t] deny and tell Ivy what to do instead
```

---

## Tool Runtime

| Tool | Notes |
|---|---|
| `terminal`, `process` | PowerShell / Bash (foreground & `background=true`), guarded by `SafetyApprovalPolicy` |
| `read_file`, `write_file`, `list_directory`, `patch_file`, `search_files` | Workspace file operations with reversible `FileCheckpointStore` snapshots |
| `code_runner` | Isolated Python subprocess with timeout (default 30s) |
| `web_fetch`, `web_search` | Readable text extraction / result lists |
| `read_skill`, `skill_manage` | Load a `SKILL.md` on demand · create/patch/delete learned skills |
| `session_search` | SQLite FTS5 search across saved sessions |
| `todo_list` | Structured task list tracking within a session |
| `remember_preference` | Persistent user preference profile stored in SQLite |
| `clarify` | Interactive multi-choice or custom clarifying question prompt |
| `cron`, `delegate_task`, MCP tools | Conditional scheduler, isolated subagent runner, and `.mcp.json` stdio MCP tools |

---

## Project Layout

```
ivy/
├── core/      # agent loop, LLM adapter, prompt & redaction, subagent, types
├── tools/     # terminal, process_registry, files, checkpoint, runner, web, registry, approval, mcp, todo, clarify, profile
├── memory/    # SQLite store, FTS5, context window, profile
├── skills/    # SKILL.md engine, generator, read/manage tools
├── gateway/   # server, session hub, turn lease, adapters (cli/telegram/discord)
├── cron/      # scheduler & task model
└── tui/       # app, commands, prompt, picker, views, theme, helpers, doctor
skills/        # learned SKILL.md files
tests/         # pytest suite
AGENTS.md      # agent architecture & repo conventions
```

---

## Testing

```bash
uv run pytest -v
```

The suite covers the agent loop, LLM routing, tools, memory/FTS, skills, cron, gateway, CLI, and an end-to-end integration path.

# Ivy

Autonomous, self-improving Python agent inspired by **Hermes Agent**. Built cleanly for Python 3.11+ (verified on Python 3.15.2) and managed with Astral `uv`.

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
uv run ivy --continue          # resume the most recently updated session
uv run ivy --theme ocean
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

### Skills & Cron from the CLI

```bash
uv run ivy skills list
uv run ivy skills inspect coding-assistant
uv run ivy skills remove <name>
uv run ivy cron list
uv run ivy model              # print the active model
```

---

## TUI Commands

Type `/` for autocomplete. Prefixes and aliases resolve automatically.

| Command | Action |
|---|---|
| `/setup` | Configure API base URL, API key and model (writes to `.env`) |
| `/model` | Switch model — picker, or `/model <name>` |
| `/session` · `/resume` | Session picker; `/session list`, `/session <id\|#N>` |
| `/new` · `/fork` · `/title` | New session · branch current session · set/view title |
| `/undo` · `/retry` | Roll back last turn · roll back and re-send the prompt |
| `/history` · `/save` · `/copy` | Show turns · export (`md`/`json`) · copy last reply |
| `/status` · `/context` · `/usage` | Session/approval state · token breakdown · API call totals |
| `/compact` | LLM-summarise older turns to reclaim context |
| `/diff` | `git diff` in the workspace (`/diff staged`, `/diff --stat`) |
| `/verbose` · `/focus` | Cycle compact/verbose/focus · toggle tool-line hiding |
| `/yolo` | Toggle auto-approval of destructive commands for this session |
| `/prompt` | Compose a multiline prompt in `$EDITOR` |
| `/skills` · `/cron` · `/memory` | List/inspect/remove/reflect · list tasks · FTS5 search |
| `/theme` · `/clear` · `/help` · `/exit` | Theme picker · clear session · help · quit |

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
```

Files are capped at ~12 KB each to protect the context window.

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
| `terminal` | PowerShell (Windows) / Bash, guarded by `SafetyApprovalPolicy` |
| `read_file`, `write_file`, `list_directory`, `patch_file`, `search_files` | Workspace file operations |
| `code_runner` | Isolated Python subprocess with timeout (default 30s) |
| `web_fetch`, `web_search` | Readable text extraction / result lists |
| `read_skill` | Load a `SKILL.md` on demand |

---

## Project Layout

```
ivy/
├── core/      # agent loop, LLM adapter, prompt, subagent, types
├── tools/     # terminal, files, runner, web, registry, approval
├── memory/    # SQLite store, FTS5, context window, profile
├── skills/    # SKILL.md engine, generator, read tool
├── gateway/   # server, session hub, turn lease, adapters (cli/telegram/discord)
├── cron/      # scheduler & task model
└── tui/       # app, prompt, picker, views, theme, helpers
skills/        # learned SKILL.md files
tests/         # pytest suite
```

---

## Testing

```bash
uv run pytest -v
```

The suite covers the agent loop, LLM routing, tools, memory/FTS, skills, cron, gateway, CLI, and an end-to-end integration path.

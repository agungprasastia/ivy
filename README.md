# Aether Agent

Autonomous self-improving Python agent inspired by **Hermes Agent**, built cleanly for Python 3.11+ (verified on Python 3.15.2) and managed with Astral `uv`.

## Overview

Aether Agent is a modular, async-first AI agent ecosystem featuring:
- **ReAct Autonomous Execution Engine:** Iterative reasoning, tool execution, and error/rejection recovery.
- **Universal Native LLM Adapter:** Supports OpenAI, Anthropic, Gemini, OpenRouter, Ollama, and vLLM via native `openai` and `anthropic` async SDKs (prebuilt binary wheels with zero C-compiler dependencies).
- **Multi-Backend Tool Runtime & Safety Guard:** PowerShell/Bash terminal execution, file tools (`read_file`, `write_file`, `list_directory`, `patch_file`, `search_files`), isolated Python `code_runner` with timeout protection, web fetch/search, and destructive command interception (`SafetyApprovalPolicy`).
- **Closed Learning Loop (`agentskills.io`):** Dynamic `SKILL.md` discovery and automated post-task reflection (`SkillGenerator`).
- **SQLite FTS5 Memory & User Profiling:** Cross-session full-text recall (`messages_fts`) with query sanitization, persistent user preferences, and sliding context window compression.
- **Multi-Platform Messaging Gateway & Cron Scheduler:** Central async gateway daemon with fail-closed `ALLOWED_USERS` whitelist, per-session `TurnLeaseManager`, Telegram (`aiogram`) & Discord (`discord.py`) adapters, and `croniter` background scheduling.
- **Rich Terminal TUI:** Interactive prompt with slash-command autocomplete (`/model`, `/skills`, `/gateway`, `/cron`, `/memory`, `/clear`, `/help`, `/exit`).

## Installation

Requires Python 3.11+ and Astral `uv`.

```bash
# Core + CLI/TUI
uv sync

# Including Telegram & Discord gateway adapters
uv sync --extra gateway
```

## Configuration

Copy `.env.example` to `.env` and configure your provider keys and whitelist:

```bash
cp .env.example .env
```

Key environment variables:
- `DEFAULT_MODEL`: e.g. `gpt-4o-mini`, `anthropic/claude-sonnet-4-5`, `gemini/gemini-2.5-flash`, `openrouter/meta-llama/llama-3.3-70b-instruct`, `ollama/qwen2.5`
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`
- `ALLOWED_USERS`: Comma-separated user IDs allowed to control Telegram/Discord bots
- `TELEGRAM_TOKEN`, `DISCORD_TOKEN`

## Usage

Run interactive Terminal TUI:
```bash
uv run aether
```

Override model or resume a specific session:
```bash
uv run aether --model gemini/gemini-2.5-flash --session my-project
```

Manage skills from CLI:
```bash
uv run aether skills list
uv run aether skills inspect coding-assistant
```

Start the background Gateway & Cron daemon:
```bash
uv run aether gateway start
```

## Testing

```bash
uv run pytest -v
```

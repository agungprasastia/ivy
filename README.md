# Aether Agent

Autonomous self-improving Python agent inspired by Hermes Agent.

## Overview
Aether Agent is a modular, extensible AI agent ecosystem featuring:
- ReAct reasoning loop with tool calling and human-in-the-loop safety approval.
- Closed learning loop adhering to the `agentskills.io` standard (`SKILL.md`).
- SQLite storage with Full-Text Search (FTS5) for cross-session recall.
- Multi-platform messaging gateway (CLI, Telegram, Discord).
- Asynchronous background cron scheduler.
- Rich terminal TUI with syntax highlighting and slash commands.

## Installation
Requires Python 3.11+ and Astral `uv`.

```bash
uv sync
```

## Configuration
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

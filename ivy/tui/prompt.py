"""Prompt-toolkit input configuration and slash command autocomplete."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

SLASH_COMMANDS = [
    "/model",
    "/skills",
    "/gateway",
    "/cron",
    "/memory",
    "/clear",
    "/help",
    "/exit",
]


def build_slash_completer() -> WordCompleter:
    """Create autocomplete completer for slash commands."""
    return WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=True)


def create_prompt_session() -> PromptSession[str]:
    """Create an interactive prompt_toolkit session with history and slash completion."""
    return PromptSession(
        history=InMemoryHistory(),
        completer=build_slash_completer(),
        complete_while_typing=True,
    )

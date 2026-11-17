"""Prompt-toolkit input configuration, bottom status bar, and slash autocomplete."""

from __future__ import annotations

from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from ivy.tui.theme import ThemeConfig, get_theme

SLASH_COMMANDS = [
    "/model",
    "/skills",
    "/gateway",
    "/cron",
    "/memory",
    "/theme",
    "/clear",
    "/help",
    "/exit",
]

SLASH_COMMAND_META: dict[str, str] = {
    "/model": "Switch or inspect active LLM model",
    "/skills": "List, inspect, remove, or reflect skills",
    "/gateway": "Show Telegram/Discord gateway status",
    "/cron": "List scheduled cron tasks",
    "/memory": "Search past sessions via SQLite FTS5",
    "/theme": "Switch TUI skin (ivy, hermes, ocean)",
    "/clear": "Clear current session history",
    "/help": "Show command reference",
    "/exit": "Exit Ivy",
}


def build_slash_completer() -> WordCompleter:
    """Create autocomplete completer with descriptions for slash commands."""
    return WordCompleter(
        SLASH_COMMANDS,
        ignore_case=True,
        sentence=True,
        meta_dict=SLASH_COMMAND_META,
    )


def build_prompt_style(theme: ThemeConfig | None = None) -> Style:
    """Build prompt_toolkit style matching the active TUI theme."""
    t = theme or get_theme()
    return Style.from_dict(
        {
            "prompt": f"bold {t.ui_accent}",
            "bottom-toolbar": f"bg:{t.status_bar_bg} {t.status_bar_text}",
            "bottom-toolbar.strong": f"bg:{t.status_bar_bg} bold {t.status_bar_strong}",
            "bottom-toolbar.dim": f"bg:{t.status_bar_bg} {t.status_bar_dim}",
            "completion-menu": f"bg:{t.completion_bg} {t.status_bar_text}",
            "completion-menu.completion.current": f"bg:{t.completion_current_bg} bold {t.status_bar_strong}",
            "completion-menu.meta.completion": f"bg:{t.completion_bg} {t.status_bar_dim}",
            "completion-menu.meta.completion.current": f"bg:{t.completion_current_bg} {t.status_bar_text}",
        }
    )


def build_bottom_toolbar(
    model: str,
    session_id: str,
    tool_count: int = 0,
    skill_count: int = 0,
    theme: ThemeConfig | None = None,
) -> HTML:
    """Render themed bottom status bar for the prompt_toolkit input."""
    t = theme or get_theme()
    return HTML(
        f"<bottom-toolbar.strong> {t.glyph} Ivy </bottom-toolbar.strong>"
        f"<bottom-toolbar.dim> │ </bottom-toolbar.dim>"
        f"<bottom-toolbar>{model}</bottom-toolbar>"
        f"<bottom-toolbar.dim> │ </bottom-toolbar.dim>"
        f"<bottom-toolbar>session:{session_id}</bottom-toolbar>"
        f"<bottom-toolbar.dim> │ </bottom-toolbar.dim>"
        f"<bottom-toolbar.dim>{tool_count} tools · {skill_count} skills · /help · Ctrl+D exit</bottom-toolbar.dim>"
    )


def create_prompt_session(
    theme: ThemeConfig | None = None,
    bottom_toolbar: Callable[[], HTML] | None = None,
) -> PromptSession[str]:
    """Create an interactive prompt_toolkit session with themed chrome and completion."""
    return PromptSession(
        history=InMemoryHistory(),
        completer=build_slash_completer(),
        complete_while_typing=True,
        style=build_prompt_style(theme),
        bottom_toolbar=bottom_toolbar,
    )

"""Prompt-toolkit input configuration, bottom status bar, and slash autocomplete."""

from __future__ import annotations

import shutil
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import FormattedText
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
    """Build prompt_toolkit style matching the active TUI theme (noreverse footer)."""
    t = theme or get_theme()
    return Style.from_dict(
        {
            "prompt": f"bold {t.ui_accent}",
            "bottom-toolbar": f"noreverse bg:default {t.status_bar_text}",
            "bottom-toolbar.text": f"noreverse bg:default {t.status_bar_text}",
            "bottom-toolbar.rule": f"noreverse bg:default {t.status_bar_dim}",
            "bottom-toolbar.strong": f"noreverse bg:default bold {t.status_bar_strong}",
            "bottom-toolbar.accent": f"noreverse bg:default {t.ui_accent}",
            "bottom-toolbar.dim": f"noreverse bg:default {t.status_bar_dim}",
            "completion-menu": f"noreverse bg:{t.completion_bg} {t.status_bar_text}",
            "completion-menu.completion.current": (
                f"noreverse bg:{t.completion_current_bg} bold {t.status_bar_strong}"
            ),
            "completion-menu.meta.completion": f"noreverse bg:{t.completion_bg} {t.status_bar_dim}",
            "completion-menu.meta.completion.current": (
                f"noreverse bg:{t.completion_current_bg} {t.status_bar_text}"
            ),
        }
    )


def build_bottom_toolbar(
    model: str,
    session_id: str,
    tool_count: int = 0,
    skill_count: int = 0,
    theme: ThemeConfig | None = None,
    width: int | None = None,
) -> FormattedText:
    """Render Hermes-style two-row clean footer (top rule + split left/right status)."""
    t = theme or get_theme()
    cols = max(40, width or shutil.get_terminal_size((100, 24)).columns)

    left_parts: list[tuple[str, str]] = [
        ("class:bottom-toolbar.strong", f" {t.glyph} Ivy"),
        ("class:bottom-toolbar.dim", " · "),
        ("class:bottom-toolbar.text", model),
        ("class:bottom-toolbar.dim", " · "),
        ("class:bottom-toolbar.text", session_id),
        ("class:bottom-toolbar.dim", f" · {tool_count} tools · {skill_count} skills"),
    ]

    right_parts: list[tuple[str, str]] = [
        ("class:bottom-toolbar.strong", "/help"),
        ("class:bottom-toolbar.dim", " commands · "),
        ("class:bottom-toolbar.strong", "/theme"),
        ("class:bottom-toolbar.dim", " skin · "),
        ("class:bottom-toolbar.strong", "Ctrl+D"),
        ("class:bottom-toolbar.dim", " exit "),
    ]

    left_len = sum(len(txt) for _, txt in left_parts)
    right_len = sum(len(txt) for _, txt in right_parts)

    fragments: list[tuple[str, str]] = [
        ("class:bottom-toolbar.rule", "─" * cols),
        ("", "\n"),
    ]
    if left_len + right_len + 2 <= cols:
        pad = " " * (cols - left_len - right_len)
        fragments.extend(left_parts)
        fragments.append(("class:bottom-toolbar.dim", pad))
        fragments.extend(right_parts)
    else:
        fragments.extend(left_parts)

    return FormattedText(fragments)


def create_prompt_session(
    theme: ThemeConfig | None = None,
    bottom_toolbar: Callable[[], FormattedText] | None = None,
) -> PromptSession[str]:
    """Create an interactive prompt_toolkit session with themed chrome and completion."""
    return PromptSession(
        history=InMemoryHistory(),
        completer=build_slash_completer(),
        complete_while_typing=True,
        style=build_prompt_style(theme),
        bottom_toolbar=bottom_toolbar,
    )
